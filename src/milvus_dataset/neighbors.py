"""
Neighbors module provides functionality for computing nearest neighbors in vector spaces.

This module contains classes and utilities for computing and managing nearest neighbor
relationships between vectors, supporting both CPU and GPU computations when available.
"""

__all__ = [
    "NeighborsComputation",
    "TempFolderManager",
]

import concurrent.futures
import time
import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import Tuple

import numba as nb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from .log_config import logger

try:
    import cupy as cp
    from cuvs.distance import pairwise_distance as cuvs_pairwise_distance

    GPU_AVAILABLE = True
except ImportError as e:
    logger.info(f"import failed with error {e}")
    GPU_AVAILABLE = False


@nb.njit(parallel=True)
def process_neighbors_fast(ids: np.ndarray, distances: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast processing of separate id and distance arrays using Numba.

    Args:
        ids: numpy array containing neighbor ids
        distances: numpy array containing corresponding distances
        top_k: number of top neighbors to keep

    Returns:
        Tuple[np.ndarray, np.ndarray]: Sorted arrays for ids and distances
    """
    n = ids.shape[0]  # number of rows
    m = min(ids.shape[1], top_k)  # number of columns to keep

    # Pre-allocate output arrays
    final_ids = np.empty((n, m), dtype=np.int64)
    final_distances = np.empty((n, m), dtype=np.float64)

    # Process each row in parallel
    for i in nb.prange(n):
        # Get sort indices for this row
        sort_idx = np.argsort(distances[i, :])[:m]

        # Store sorted results
        final_ids[i] = ids[i, sort_idx]
        final_distances[i] = distances[i, sort_idx]

    return final_ids, final_distances


def parallel_read_parquet(file_path: str, fs, pk_field_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel reading of parquet files.
    """
    with fs.open(file_path, "rb") as f:
        df = pq.read_table(f).to_pandas()
    return np.array(df[pk_field_name].tolist()), np.array(df["neighbors_id"].tolist())



class TempFolderManager:
    """Manages temporary folders for neighbor computation results.

    This class handles the creation and management of temporary folders
    needed during the neighbor computation process.

    Args:
        neighbors (Dataset): The neighbors dataset to manage temporary folders for
    """

    def __init__(self, neighbors: "Dataset") -> None:
        """Initialize the TempFolderManager.

        Args:
            neighbors (Dataset): The neighbors dataset instance
        """
        self.neighbors = neighbors
        self.base_tmp_path = (
            f"{self.neighbors.root_path}/{self.neighbors.name}/{self.neighbors.split}"
        )

    def ensure_dir(self, path: str) -> int:
        """Ensure a directory exists, creating it if necessary.

        Args:
            path (str): The directory path to ensure exists

        Returns:
            int: Number of parquet files in the directory
        """
        self.neighbors.fs.makedirs(path, exist_ok=True)
        try:
            # 只统计parquet文件的数量
            parquet_files = self.neighbors.fs.glob(f"{path}/*.parquet")
            return len(parquet_files)
        except FileNotFoundError:
            return 0

    @contextmanager
    def temp_folder(self, folder_name: str) -> Generator[str, None, None]:
        """Create and manage a temporary folder.

        Args:
            folder_name (str): Name of the temporary folder

        Yields:
            str: Path to the temporary folder
        """
        tmp_path = f"{self.base_tmp_path}/{folder_name}"
        try:
            # 创建临时文件夹并确保它存在
            self.ensure_dir(tmp_path)
            logger.debug(f"Created temporary folder: {tmp_path}")
            yield tmp_path
        finally:
            # 在退出上下文时删除临时文件夹
            if self.neighbors.fs.exists(tmp_path):
                logger.debug(f"Removing temporary folder: {tmp_path}")
                self.neighbors.fs.rm(tmp_path, recursive=True)


class NeighborsComputation:
    """Computes nearest neighbors for vector data.

    This class handles the computation of nearest neighbors for large-scale
    vector datasets, supporting both CPU and GPU acceleration when available.

    Args:
        dataset_dict (Dict[str, Dataset]): Dictionary containing dataset information
        vector_field_name (str): Name of the field containing vector data
        pk_field_name (str): Name of the primary key field (default: "id")
        query_expr (Optional[str]): Optional query expression for filtering data
        top_k (int): Number of nearest neighbors to compute (default: 1000)
        metric_type (str): Distance metric to use (default: "cosine")
        max_rows_per_epoch (int): Maximum rows to process per epoch (default: 1000000)
        test_batch_size (int): Batch size for test data processing (default: 5000)
    """

    def __init__(
        self,
        dataset_dict: dict[str, "Dataset"],
        vector_field_name: str,
        pk_field_name: str = "id",
        query_expr: str | None = None,
        top_k: int = 1000,
        metric_type: str = "cosine",
        max_rows_per_epoch: int = 30000,
        test_batch_size: int = 5000,
    ) -> None:
        """Initialize the NeighborsComputation instance.

        Args:
            dataset_dict (Dict[str, Dataset]): Dictionary containing dataset information
            vector_field_name (str): Name of the field containing vector data
            pk_field_name (str): Name of the primary key field (default: "id")
            query_expr (Optional[str]): Optional query expression for filtering data
            top_k (int): Number of nearest neighbors to compute (default: 1000)
            metric_type (str): Distance metric to use (default: "cosine")
            max_rows_per_epoch (int): Maximum rows to process per epoch (default: 1000000)
            test_batch_size (int): Batch size for test data processing (default: 5000)
        """
        self.dataset_dict = dataset_dict
        self.vector_field_name = vector_field_name
        self.pk_field_name = pk_field_name
        self.query_expr = query_expr
        self.top_k = top_k
        self.metric_type = metric_type
        self.max_rows_per_epoch = max_rows_per_epoch
        self.test_batch_size = test_batch_size
        self.neighbors = self.dataset_dict["neighbors"]
        self.file_name = f"{self.neighbors.root_path}/{self.neighbors.name}/{self.neighbors.split}/neighbors-vector-{vector_field_name}-pk-{pk_field_name}-expr-{self.query_expr}-metric-{metric_type}.parquet"

    def _calculate_num_epochs(self) -> int:
        """Calculate the number of epochs needed for computation.

        Returns:
            int: Number of epochs
        """
        total_rows = self.dataset_dict["train"].get_total_rows("train")
        return max(1, (total_rows + self.max_rows_per_epoch - 1) // self.max_rows_per_epoch)

    @staticmethod
    @nb.njit("int64[:,::1](float32[:,::1])", parallel=True)
    def fast_sort(a: np.ndarray) -> np.ndarray:
        """Perform fast sorting of an array.

        Args:
            a (np.ndarray): Input array of shape (n, m)

        Returns:
            np.ndarray: Sorted indices array of shape (n, m)
        """
        b = np.empty(a.shape, dtype=np.int64)
        for i in nb.prange(a.shape[0]):
            b[i, :] = np.argsort(a[i, :])
        return b

    def compute_neighbors(
        self,
        test_data: pd.DataFrame,
        train_data: pd.DataFrame,
        vector_field_name: str,
        tmp_path: str,
    ) -> None:
        """Compute nearest neighbors for a batch of test data.

        Args:
            test_data (pd.DataFrame): Test data batch
            train_data (pd.DataFrame): Train data batch
            vector_field_name (str): Name of the field containing vector data
            tmp_path (str): Temporary path for storing intermediate results
        """
        test_emb = np.array(test_data[vector_field_name].tolist())
        train_emb = np.array(train_data[vector_field_name].tolist())

        test_idx = test_data[self.pk_field_name].tolist()
        train_idx = train_data[self.pk_field_name].tolist()

        t0 = time.time()

        if GPU_AVAILABLE:
            logger.info("Using GPU for neighbor computation")
            test_emb_gpu = cp.array(test_emb, dtype=cp.float32)
            train_emb_gpu = cp.array(train_emb, dtype=cp.float32)
            distance = cuvs_pairwise_distance(
                train_emb_gpu, test_emb_gpu, metric=self.metric_type
            )
            distance = cp.asnumpy(distance)
            distance = np.array(distance.T, order="C")
            distance_sorted_arg = self.fast_sort(distance)
            indices = distance_sorted_arg[:, : self.top_k]
            distances = np.array([distance[i, indices[i]] for i in range(len(indices))])

        else:
            logger.info("Using CPU for neighbor computation")
            logger.info(f"test_emb shape: {test_emb.shape}, train_emb shape: {train_emb.shape}")
            if self.metric_type == "inner_product":
                # Compute inner product using matrix multiplication

                distance = -1 * (train_emb @ test_emb.T)  # Transpose to get (num_test, num_train)

            else:
                distance = pairwise_distances(train_emb, Y=test_emb, metric=self.metric_type, n_jobs=-1)
            logger.info(f"distance matrix shape: {distance.shape}")
            distance = np.array(distance.T, order="C", dtype=np.float32)
            distance_sorted_arg = self.fast_sort(distance)
            indices = distance_sorted_arg[:, : self.top_k]
            distances = np.array([distance[i, indices[i]] for i in range(len(indices))])

        logger.info(f"Neighbor computation cost time: {time.time() - t0}")

        result = np.empty(
            indices.shape, dtype=[(self.pk_field_name, "int64"), ("distance", "float64")]
        )
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                result[i, j] = (train_idx[indices[i, j]], distances[i, j])

        df_neighbors = pd.DataFrame({self.pk_field_name: test_idx, "neighbors_id": result.tolist()})
        logger.info(f"Writing neighbors to {tmp_path}")
        # 使用TempFolderManager的ensure_dir方法
        temp_manager = TempFolderManager(self.neighbors)
        file_num = temp_manager.ensure_dir(tmp_path)
        file_name = f"{tmp_path}/neighbors_{file_num}.parquet"
        logger.info(f"Writing neighbors to {file_name}")
        with self.neighbors.fs.open(file_name, "wb") as f:
            df_neighbors.to_parquet(f, engine="pyarrow", compression="snappy")

    def merge_neighbors(self, final_file_name: str | None = None, tmp_path: str | None = None) -> str:
        """Merge intermediate neighbor results with separate id and distance handling."""
        t_start = time.time()
        file_list = self.neighbors.fs.glob(f"{tmp_path}/*.parquet")
        logger.info(f"Starting parallel file reading for {len(file_list)} files...")

        # Parallel file reading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(parallel_read_parquet, f, self.neighbors.fs, self.pk_field_name)
                for f in file_list
            ]
            results = list(tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Reading files"
            ))

        # Combine results and separate ids and distances
        test_idx = results[0].result()[0]  # Use first file's test_idx
        neighbors_arrays = [f.result()[1] for f in results]

        # Extract ids and distances from the structured arrays
        total_neighbors = sum(arr.shape[1] for arr in neighbors_arrays)
        ids = np.empty((len(test_idx), total_neighbors), dtype=np.int64)
        distances = np.empty((len(test_idx), total_neighbors), dtype=np.float64)

        current_col = 0
        for arr in neighbors_arrays:
            cols = arr.shape[1]
            for i in range(len(test_idx)):
                for j in range(cols):
                    ids[i, current_col + j] = arr[i, j][0]  # id
                    distances[i, current_col + j] = arr[i, j][1]  # distance
            current_col += cols

        logger.info(f"File reading and merging completed in {time.time() - t_start:.3f}s")

        # Process and sort neighbors using separate arrays
        t_process = time.time()
        final_ids, final_distances = process_neighbors_fast(ids, distances, self.top_k)
        logger.info(f"Processing and sorting completed in {time.time() - t_process:.3f}s")

        # # Create final structured array for the DataFrame
        # neighbors_result = np.empty(final_ids.shape, dtype=[('id', 'int64'), ('distance', 'float64')])
        # for i in range(final_ids.shape[0]):
        #     for j in range(final_ids.shape[1]):
        #         neighbors_result[i, j] = (final_ids[i, j], final_distances[i, j])

        # Create DataFrame efficiently
        t_df = time.time()
        df = pd.DataFrame({
            "idx": test_idx,
            "neighbors_id": final_ids.tolist(),
            "neighbors_distance": final_distances.tolist(),
            "metric": self.metric_type,
            "query_expr": self.query_expr,
            "pk_field_name": self.pk_field_name,
            "vector_field_name": self.vector_field_name,
            "top_k": self.top_k
        })
        logger.info(f"DataFrame creation completed in {time.time() - t_df:.3f}s")

        # Write results
        t_write = time.time()
        with self.neighbors.fs.open(final_file_name, "wb") as f:
            df.to_parquet(
                f,
                engine="pyarrow",
                compression="snappy",
                use_dictionary=False,
                row_group_size=100000
            )
        logger.info(f"File writing completed in {time.time() - t_write:.3f}s")

        return final_file_name

    def merge_final_results(self, partial_files: list[str]) -> None:
        """Merge all partial results into a single file.

        Args:
            partial_files (List[str]): List of partial result files
        """
        logger.info("Merging all partial results into a single file")
        t0 = time.time()

        def read_partial_file(file_name: str) -> pd.DataFrame:
            """Read a partial result file.

            Args:
                file_name (str): Partial result file name

            Returns:
                pd.DataFrame: Partial result data frame
            """
            with self.neighbors.fs.open(file_name, "rb") as f:
                return pq.read_table(f).to_pandas()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            dfs = list(
                tqdm(
                    executor.map(read_partial_file, partial_files),
                    total=len(partial_files),
                )
            )

        final_df = pd.concat(dfs, ignore_index=True)
        final_df = final_df.sort_values(self.pk_field_name).reset_index(drop=True)

        final_file_name = self.file_name
        logger.info(f"Writing final merged results to {final_file_name}")

        with self.neighbors.fs.open(final_file_name, "wb") as f:
            final_df.to_parquet(f, engine="pyarrow", compression="snappy")

        logger.info(f"Final merge completed. Total time: {time.time() - t0}")

        # Clean up partial files
        for file in partial_files:
            self.neighbors.fs.rm(file)
        logger.info("Cleaned up partial result files")

    def compute_ground_truth(self):
        logger.info(f"Computing ground truth")
        start_time = time.time()

        # Get total counts directly
        total_test_rows = len(self.dataset_dict['test'])
        total_train_rows = len(self.dataset_dict['train'])

        # Calculate expected number of batches using math.ceil
        test_count = math.ceil(total_test_rows / self.test_batch_size)
        train_count = math.ceil(total_train_rows / self.max_rows_per_epoch)

        logger.info(f"Total test batches: {test_count}, total test rows: {total_test_rows}")
        logger.info(f"Total train batches: {train_count}, total train rows: {total_train_rows}")

        test_data_generator = self.dataset_dict['test'].read(mode='batch', batch_size=self.test_batch_size)
        train_data_generator = self.dataset_dict['train'].read(mode='batch', batch_size=self.max_rows_per_epoch)

        temp_manager = TempFolderManager(self.neighbors)
        partial_files = []
        processed_test_rows = 0

        with temp_manager.temp_folder("tmp") as tmp_path:
            for i, test_data in enumerate(test_data_generator):
                batch_start_time = time.time()
                processed_test_rows += len(test_data)
                progress = (processed_test_rows / total_test_rows) * 100
                elapsed_time = time.time() - start_time
                eta = (elapsed_time / processed_test_rows) * (total_test_rows - processed_test_rows) if processed_test_rows > 0 else 0

                logger.info(f"Processing test batch {i+1}/{test_count} ({progress:.2f}% complete)")
                logger.info(f"Test batch size: {len(test_data)}, Elapsed: {elapsed_time:.2f}s, ETA: {eta:.2f}s")

                with temp_manager.temp_folder(f"tmp_{i}") as tmp_test_split_path:
                    processed_train_rows = 0
                    for j, train_train in enumerate(train_data_generator):
                        processed_train_rows += len(train_train)
                        train_progress = (processed_train_rows / total_train_rows) * 100
                        logger.info(f"Computing neighbors for train batch {j+1}/{train_count} ({train_progress:.2f}% of train data)")
                        logger.info(f"Train batch size: {len(train_train)}")
                        self.compute_neighbors(test_data, train_train, self.vector_field_name, tmp_test_split_path)

                    # Reset train data generator for next test batch
                    train_data_generator = self.dataset_dict['train'].read(mode='batch', batch_size=self.max_rows_per_epoch)

                    merged_file_name = f"{tmp_path}/neighbors-{self.query_expr}-{i}.parquet"
                    partial_file = self.merge_neighbors(merged_file_name, tmp_test_split_path)
                    partial_files.append(partial_file)

                batch_time = time.time() - batch_start_time
                logger.info(f"Completed test batch {i+1} in {batch_time:.2f}s")

            total_time = time.time() - start_time
            logger.info(f"All test batches processed in {total_time:.2f}s")
            self.merge_final_results(partial_files)

        final_time = time.time() - start_time
        logger.info(f"Ground truth computation completed in {final_time:.2f}s")
