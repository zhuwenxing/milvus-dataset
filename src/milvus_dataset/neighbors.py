import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import pairwise_distances
from .logging import logger
import time
import numba as nb
import os
import concurrent.futures
from tqdm import tqdm
from contextlib import contextmanager

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
    def __init__(self, dataset_dict, vector_field_name, query_expr=None, top_k: int = 1000, metric_type: str = "cosine",
                 max_rows_per_epoch: int = 1000000):
        self.dataset_dict = dataset_dict
        self.vector_field_name = vector_field_name
        self.query_expr = query_expr
        self.top_k = top_k
        self.metric_type = metric_type
        self.max_rows_per_epoch = max_rows_per_epoch
        self.neighbors = self.dataset_dict['neighbors']
        self.file_name = f"{self.neighbors.root_path}/{self.neighbors.name}/{self.neighbors.split}/neighbors-{self.query_expr}.parquet"

    def _calculate_num_epochs(self) -> int:
        total_rows = self.dataset_dict['train'].get_total_rows('train')
        return max(1, (total_rows + self.max_rows_per_epoch - 1) // self.max_rows_per_epoch)

    @staticmethod
    @nb.njit('int64[:,::1](float64[:,::1])', parallel=True)
    def fast_sort(a):
        b = np.empty(a.shape, dtype=np.int64)
        for i in nb.prange(a.shape[0]):
            b[i, :] = np.argsort(a[i, :])
        return b

    def compute_neighbors(self, test_data, train_data, vector_field_name, tmp_path):
        neighbors = self.dataset_dict['neighbors']
        test_emb = np.array(test_data[vector_field_name].tolist())
        train_emb = np.array(train_data[vector_field_name].tolist())

        distance = pairwise_distances(train_emb, Y=test_emb, metric=self.metric_type, n_jobs=-1)
        distance = np.array(distance.T, order='C')
        test_idx = test_data["id"].tolist()
        idx = train_data["id"].tolist()

        t0 = time.time()
        distance_sorted_arg = self.fast_sort(distance)
        logger.info(f"Sort cost time: {time.time() - t0}")

        top_k = distance_sorted_arg[:, :self.top_k]
        result = np.empty(top_k.shape, dtype=[('idx', "i8"), ('distance', "f8")])
        for index, value in np.ndenumerate(top_k):
            result[index] = (idx[value], distance[index[0], value])

        df_neighbors = pd.DataFrame({
            "id": test_idx,
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
            test_idx = np.array(df_n["id"].tolist())
            tmp_neighbors_id = np.array(df_n["neighbors_id"].tolist())
            if neighbors_id is None:
                neighbors_id = tmp_neighbors_id
            else:
                neighbors_id = np.concatenate((neighbors_id, tmp_neighbors_id), axis=1)
        result = np.empty(neighbors_id.shape, dtype=[('idx', "i8"), ('distance', "f8")])
        for index, value in np.ndenumerate(neighbors_id):
            result[index] = (neighbors_id[index][0], neighbors_id[index][1])
        logger.info(f"result \n: {result}")
        sorted_result = np.sort(result, axis=1, order=["distance"])
        final_result = np.empty(sorted_result.shape, dtype="i8")
        for index, value in np.ndenumerate(sorted_result):
            final_result[index] = sorted_result[index][0]
        logger.info(f"final_result \n: {final_result}")
        df = pd.DataFrame(data={
            "id": test_idx,
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
        final_df = final_df.sort_values('id').reset_index(drop=True)

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

        test_data_batches = list(self.dataset_dict['test'].read(mode='batch', batch_size=1000))
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


