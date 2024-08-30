import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import pairwise_distances

from .log_config import logger
import time
import numba as nb
import os
import concurrent.futures
from tqdm import tqdm
from contextlib import contextmanager

try:
    from pylibraft.common import Handle
    from pylibraft.distance import pairwise_distance as raft_pairwise_distance
    import cupy as cp
    from pylibraft.neighbors.brute_force import knn
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class TempFolderManager:
    def __init__(self, neighbors):
        self.neighbors = neighbors
        self.base_tmp_path = f"{self.neighbors.root_path}/{self.neighbors.name}/{self.neighbors.split}"

    @contextmanager
    def temp_folder(self, folder_name):
        tmp_path = f"{self.base_tmp_path}/{folder_name}"
        try:
            # 创建临时文件夹
            self.neighbors.fs.makedirs(tmp_path, exist_ok=True)
            logger.debug(f"Created temporary folder: {tmp_path}")
            yield tmp_path
        finally:
            # 在退出上下文时删除临时文件夹
            if self.neighbors.fs.exists(tmp_path):
                logger.debug(f"Removing temporary folder: {tmp_path}")
                self.neighbors.fs.rm(tmp_path, recursive=True)


class NeighborsComputation:
    def __init__(self, dataset_dict, vector_field_name, pk_field_name="id", query_expr=None, top_k: int = 1000, metric_type: str = "cosine",
                 max_rows_per_epoch: int = 1000000):
        self.dataset_dict = dataset_dict
        self.vector_field_name = vector_field_name
        self.pk_field_name = pk_field_name
        self.query_expr = query_expr
        self.top_k = top_k
        self.metric_type = metric_type
        self.max_rows_per_epoch = max_rows_per_epoch
        self.neighbors = self.dataset_dict['neighbors']
        self.file_name = f"{self.neighbors.root_path}/{self.neighbors.name}/{self.neighbors.split}/neighbors-expr-{self.query_expr}-metric-{metric_type}.parquet"

    def _calculate_num_epochs(self) -> int:
        total_rows = self.dataset_dict['train'].get_total_rows('train')
        return max(1, (total_rows + self.max_rows_per_epoch - 1) // self.max_rows_per_epoch)

    @staticmethod
    @nb.njit('int64[:,::1](float32[:,::1])', parallel=True)
    def fast_sort(a):
        b = np.empty(a.shape, dtype=np.int64)
        for i in nb.prange(a.shape[0]):
            b[i, :] = np.argsort(a[i, :])
        return b

    def compute_neighbors(self, test_data, train_data, vector_field_name, tmp_path):
        test_emb = np.array(test_data[vector_field_name].tolist())
        train_emb = np.array(train_data[vector_field_name].tolist())

        test_idx = test_data[self.pk_field_name].tolist()
        train_idx = train_data[self.pk_field_name].tolist()

        t0 = time.time()

        if GPU_AVAILABLE:
            logger.info("Using GPU for neighbor computation")
            test_emb_gpu = cp.array(test_emb, dtype=cp.float32)
            train_emb_gpu = cp.array(train_emb, dtype=cp.float32)

            if self.top_k <= 1024:
                distances, indices = knn(train_emb_gpu, test_emb_gpu, k=self.top_k, metric=self.metric_type)

                distances = cp.asnumpy(distances)
                indices = cp.asnumpy(indices)
            else:
                handle = Handle()
                distance = raft_pairwise_distance(train_emb_gpu, test_emb_gpu, metric=self.metric_type, handle=handle)
                handle.sync()
                distance = cp.asnumpy(distance)
                distance = np.array(distance.T, order='C')
                distance_sorted_arg = self.fast_sort(distance)
                indices = distance_sorted_arg[:, :self.top_k]
                distances = np.array([distance[i, indices[i]] for i in range(len(indices))])

        else:
            logger.info("Using CPU for neighbor computation")
            test_emb_cpu = np.array(test_emb, dtype=np.float32)
            train_emb_cpu = np.array(train_emb, dtype=np.float32)
            distance = pairwise_distances(train_emb, Y=test_emb, metric=self.metric_type, n_jobs=-1)
            distance = np.array(distance.T, order='C', dtype=np.float32)
            distance_sorted_arg = self.fast_sort(distance)
            indices = distance_sorted_arg[:, :self.top_k]
            distances = np.array([distance[i, indices[i]] for i in range(len(indices))])

        logger.info(f"Neighbor computation cost time: {time.time() - t0}")

        result = np.empty(indices.shape, dtype=[(self.pk_field_name, "i8"), ('distance', "f8")])
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                result[i, j] = (train_idx[indices[i, j]], distances[i, j])

        df_neighbors = pd.DataFrame({
            self.pk_field_name: test_idx,
            "neighbors_id": result.tolist()
        })

        file_name = f"{tmp_path}/neighbors_{len(self.neighbors.fs.ls(tmp_path))}.parquet"
        logger.info(f"Writing neighbors to {file_name}")
        with self.neighbors.fs.open(file_name, 'wb') as f:
            df_neighbors.to_parquet(f, engine='pyarrow', compression='snappy')

    def merge_neighbors(self, final_file_name=None, tmp_path=None):
        neighbors = self.dataset_dict['neighbors']
        file_list = neighbors.fs.glob(f"{tmp_path}/*.parquet")
        logger.info(f"Found {len(file_list)} in {tmp_path}")
        neighbors_id = None
        test_idx = None
        t0 = time.time()
        for f in file_list:
            with neighbors.fs.open(f, 'rb') as f:
                df_n = pq.read_table(f).to_pandas()
            test_idx = np.array(df_n[self.pk_field_name].tolist())
            tmp_neighbors_id = np.array(df_n["neighbors_id"].tolist())
            if neighbors_id is None:
                neighbors_id = tmp_neighbors_id
            else:
                neighbors_id = np.concatenate((neighbors_id, tmp_neighbors_id), axis=1)
        result = np.empty(neighbors_id.shape, dtype=[(self.pk_field_name, "i8"), ('distance', "f8")])
        for index, value in np.ndenumerate(neighbors_id):
            result[index] = (neighbors_id[index][0], neighbors_id[index][1])
        logger.info(f"result \n: {result}")
        sorted_result = np.sort(result, axis=1, order=["distance"])
        final_result = np.empty(sorted_result.shape, dtype="i8")
        for index, value in np.ndenumerate(sorted_result):
            final_result[index] = sorted_result[index][0]
        logger.info(f"final_result \n: {final_result}")
        df = pd.DataFrame(data={
            self.pk_field_name: test_idx,
            "neighbors_id": final_result[:, :self.top_k].tolist()
        })
        logger.info(f"Writing neighbors to {final_file_name}")

        with neighbors.fs.open(final_file_name, 'wb') as f:
            df.to_parquet(f, engine='pyarrow', compression='snappy')
        logger.info(f"Merge cost time: {time.time() - t0}")
        return final_file_name

    def merge_final_results(self, partial_files):
        logger.info("Merging all partial results into a single file")
        t0 = time.time()

        def read_partial_file(file_name):
            with self.neighbors.fs.open(file_name, 'rb') as f:
                return pq.read_table(f).to_pandas()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            dfs = list(tqdm(executor.map(read_partial_file, partial_files), total=len(partial_files)))

        final_df = pd.concat(dfs, ignore_index=True)
        final_df = final_df.sort_values(self.pk_field_name).reset_index(drop=True)

        final_file_name = self.file_name
        logger.info(f"Writing final merged results to {final_file_name}")

        with self.neighbors.fs.open(final_file_name, 'wb') as f:
            final_df.to_parquet(f, engine='pyarrow', compression='snappy')

        logger.info(f"Final merge completed. Total time: {time.time() - t0}")

        # Clean up partial files
        for file in partial_files:
            self.neighbors.fs.rm(file)
        logger.info("Cleaned up partial result files")

    def compute_ground_truth(self):
        logger.info(f"Computing ground truth")

        test_data_batches = list(self.dataset_dict['test'].read(mode='batch', batch_size=2000))
        train_data_batches = list(self.dataset_dict['train'].read(mode='batch', batch_size=self.max_rows_per_epoch))
        logger.info(f"train data batches num: {len(train_data_batches)}")

        temp_manager = TempFolderManager(self.neighbors)
        partial_files = []
        with temp_manager.temp_folder("tmp") as tmp_path:
            for i, test_data in enumerate(test_data_batches):
                logger.info(f"Computing ground truth for batch, test size: {len(test_data)}")
                with temp_manager.temp_folder(f"tmp_{i}") as tmp_test_split_path:
                    for j, train_train in enumerate(train_data_batches):
                        logger.info(f"Computing ground truth for batch, train size: {len(train_train)}")
                        self.compute_neighbors(test_data, train_train, self.vector_field_name, tmp_test_split_path)

                    merged_file_name = f"{tmp_path}/neighbors-{self.query_expr}-{i}.parquet"
                    partial_file = self.merge_neighbors(merged_file_name, tmp_test_split_path)
                    partial_files.append(partial_file)
            self.merge_final_results(partial_files)


