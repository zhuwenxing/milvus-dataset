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
import math
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

if TYPE_CHECKING:
    from .core import Dataset

from .log_config import logger

try:
    import cupy
    import cupy as cp
    from cuvs.neighbors import brute_force

    GPU_AVAILABLE = True
except Exception as e:
    import_error = e
    logger.info(f"import failed with error {e}")
    GPU_AVAILABLE = False


@nb.njit(parallel=True)
def process_neighbors_fast(
    ids: np.ndarray, distances: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
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


def parallel_read_parquet(file_path: str, fs, pk_field_name: str) -> tuple[np.ndarray, np.ndarray]:
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
    def temp_folder(self, folder_name: str, persistent: bool = False) -> Generator[str, None, None]:
        """Create and manage a temporary folder.

        Args:
            folder_name (str): Name of the temporary folder
            persistent (bool): If True, don't auto-delete the folder for debugging/resume

        Yields:
            str: Path to the temporary folder
        """
        tmp_path = f"{self.base_tmp_path}/{folder_name}"
        try:
            # 创建临时文件夹并确保它存在
            self.ensure_dir(tmp_path)
            if persistent:
                logger.info(f"Created persistent folder: {tmp_path}")
            else:
                logger.debug(f"Created temporary folder: {tmp_path}")
            yield tmp_path
        finally:
            # 只在非持久化模式下删除临时文件夹
            if not persistent and self.neighbors.fs.exists(tmp_path):
                logger.debug(f"Removing temporary folder: {tmp_path}")
                self.neighbors.fs.rm(tmp_path, recursive=True)
            elif persistent:
                logger.info(f"Keeping persistent folder: {tmp_path}")


class NeighborsComputation:
    """Computes nearest neighbors for vector data.

    This class handles the computation of nearest neighbors for large-scale
    vector datasets, supporting both CPU and GPU acceleration when available.
    Implements GPU memory optimizations including train data caching and async transfers.

    Args:
        dataset_dict (Dict[str, Dataset]): Dictionary containing dataset information
        vector_field_name (str): Name of the field containing vector data
        pk_field_name (str): Name of the primary key field (default: "id")
        query_expr (Optional[str]): Optional query expression for filtering data
        top_k (int): Number of nearest neighbors to compute (default: 1000)
        metric_type (str): Distance metric to use (default: "cosine")
        max_rows_per_epoch (int): Maximum rows to process per epoch (default: 1000000)
        test_batch_size (int): Batch size for test data processing (default: 5000)
        device (str): Device to use for computation ('cpu', 'cuda', or 'auto') (default: 'auto')
    """

    def __init__(
        self,
        dataset_dict: dict[str, "Dataset"],
        vector_field_name: str,
        pk_field_name: str = "id",
        test_pk_field_name: str | None = None,
        query_expr: str | None = None,
        top_k: int = 1000,
        metric_type: str = "cosine",
        max_rows_per_epoch: int = 30000,
        test_batch_size: int = 5000,
        device: str = "auto",
    ) -> None:
        """Initialize the NeighborsComputation instance.

        Args:
            dataset_dict (Dict[str, Dataset]): Dictionary containing dataset information
            vector_field_name (str): Name of the field containing vector data
            pk_field_name (str): Name of the primary key field for train data (default: "id")
            test_pk_field_name (str, optional): Name of the primary key field for test data.
                                               If None, uses pk_field_name for both train and test data.
            query_expr (Optional[str]): Optional query expression for filtering data
            top_k (int): Number of nearest neighbors to compute (default: 1000)
            metric_type (str): Distance metric to use (default: "cosine")
            max_rows_per_epoch (int): Maximum rows to process per epoch (default: 1000000)
            test_batch_size (int): Batch size for test data processing (default: 5000)
            device (str): Device to use for computation ('cpu', 'cuda', or 'auto') (default: 'auto')
        """
        self.dataset_dict = dataset_dict
        self.vector_field_name = vector_field_name
        self.pk_field_name = pk_field_name
        self.test_pk_field_name = (
            test_pk_field_name if test_pk_field_name is not None else pk_field_name
        )
        self.query_expr = query_expr
        self.top_k = top_k
        self.metric_type = metric_type
        self.max_rows_per_epoch = max_rows_per_epoch
        self.test_batch_size = test_batch_size
        self.neighbors = self.dataset_dict["neighbors"]
        self.file_name = f"{self.neighbors.root_path}/{self.neighbors.name}/{self.neighbors.split}/neighbors-vector-{vector_field_name}-pk-{pk_field_name}-expr-{self.query_expr}-metric-{metric_type}.parquet"

        # Handle device selection
        if device not in ["cpu", "cuda", "auto"]:
            raise ValueError("Device must be one of: 'cpu', 'cuda', or 'auto'")

        self.device = device
        if device == "auto":
            self.use_gpu = GPU_AVAILABLE
        elif device == "cuda":
            if not GPU_AVAILABLE:
                raise RuntimeError(
                    f"CUDA device requested but GPU is not available with import error {import_error}"
                )
            self.use_gpu = True
        else:  # device == "cpu"
            self.use_gpu = False

        # Removed GPU caching to avoid OOM issues and low cache hit rate

    def _check_existing_results(self, tmp_path: str) -> list[str]:
        """Check for existing partial result files for resume capability.

        Args:
            tmp_path (str): Path to check for existing files

        Returns:
            list[str]: List of existing partial result files
        """
        try:
            if not self.neighbors.fs.exists(tmp_path):
                return []

            # Look for existing neighbors-test-*-train-*.parquet files
            pattern = f"{tmp_path}/neighbors-test-*-train-*.parquet"
            existing_files = self.neighbors.fs.glob(pattern)

            if existing_files:
                logger.info(f"Found {len(existing_files)} existing partial result files")
                for f in existing_files:
                    logger.info(f"  - {f}")

            return existing_files
        except Exception as e:
            logger.warning(f"Error checking existing results: {e}")
            return []

    def _load_data_optimized(self, data_df, field_name):
        """Optimized data loading with reduced memory copying.

        Args:
            data_df: DataFrame containing the data
            field_name: Name of the field to extract

        Returns:
            numpy array or cupy array depending on device
        """
        # Direct numpy array creation without intermediate list conversion
        if hasattr(data_df[field_name].iloc[0], "__iter__") and not isinstance(
            data_df[field_name].iloc[0], str
        ):
            # Vector data - use efficient numpy array creation
            data_array = np.vstack(data_df[field_name].values)
        else:
            # Scalar data
            data_array = data_df[field_name].values

        if self.use_gpu:
            # Transfer to GPU (no caching to avoid OOM)
            return cp.array(data_array, dtype=cp.float32)
        else:
            return data_array.astype(np.float32)

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

    def compute_neighbors(  # noqa
        self,
        test_data: pd.DataFrame,
        train_data: pd.DataFrame,
        vector_field_name: str,
        tmp_path: str,
        test_batch_index: int = 0,
        train_batch_index: int = 0,
    ) -> None:
        """Compute nearest neighbors for a batch of test data.
        Uses optimized GPU data loading and caching.

        Args:
            test_data (pd.DataFrame): Test data batch
            train_data (pd.DataFrame): Train data batch
            vector_field_name (str): Name of the field containing vector data
            tmp_path (str): Temporary path for storing intermediate results
        """

        def process_batch(test_batch):
            # Optimized data loading without intermediate list conversion
            test_emb = self._load_data_optimized(test_batch, vector_field_name)
            test_idx = test_batch[self.test_pk_field_name].tolist()

            if self.use_gpu:
                logger.info("Using GPU for neighbor computation")
                try:
                    # GPU computation using brute_force
                    # Map metric types to cuvs brute_force supported metrics
                    if self.metric_type == "inner_product":
                        metric_name = "inner_product"
                    elif self.metric_type == "cosine":
                        metric_name = "cosine"
                    else:
                        metric_name = "sqeuclidean"  # Default for L2/euclidean

                    # Build index fresh each time to avoid GPU memory buildup
                    index = brute_force.build(train_emb_gpu, metric=metric_name)

                    # Direct search to get neighbors and distances in one call
                    distances_gpu, indices_gpu = brute_force.search(index, test_emb, self.top_k)
                    distances = cp.asnumpy(distances_gpu)
                    indices = cp.asnumpy(indices_gpu).astype(np.int64)

                    # For inner_product, negate distances to match CPU implementation
                    # CPU uses -1 * inner_product to convert similarity to distance
                    if self.metric_type == "inner_product":
                        distances = -distances

                    return indices, distances, test_idx, True
                except (cupy.cuda.memory.OutOfMemoryError, MemoryError) as e:
                    logger.warning(f"GPU memory error occurred: {e!s}")
                    return None, None, None, False
            else:
                logger.info("Using CPU for neighbor computation")
                if self.metric_type == "inner_product":
                    distance = -1 * (train_emb @ test_emb.T)
                else:
                    distance = pairwise_distances(
                        train_emb, Y=test_emb, metric=self.metric_type, n_jobs=-1
                    )
                distance = np.array(distance.T, order="C", dtype=np.float32)
                distance_sorted_arg = self.fast_sort(distance)
                indices = distance_sorted_arg[:, : self.top_k]
                distances = np.array([distance[i, indices[i]] for i in range(len(indices))])
                return indices, distances, test_idx, True

        # Load train data optimized (no caching to avoid GPU OOM)
        train_emb_gpu = self._load_data_optimized(train_data, vector_field_name)
        # Keep CPU version for fallback
        if not self.use_gpu:
            train_emb = train_emb_gpu
        train_idx = train_data[self.pk_field_name].tolist()

        t0 = time.time()
        current_batch_size = len(test_data)
        min_batch_size = min(
            100, current_batch_size
        )  # Minimum batch size to prevent infinite loops
        logger.info(
            f"Starting neighbor computation with batch size: {current_batch_size}, min batch size: {min_batch_size}"
        )
        while current_batch_size >= min_batch_size:
            all_indices = []
            all_distances = []
            all_test_idx = []
            success = True

            for start_idx in range(0, len(test_data), current_batch_size):
                end_idx = min(start_idx + current_batch_size, len(test_data))
                test_batch = test_data.iloc[start_idx:end_idx]

                indices, distances, test_idx, batch_success = process_batch(test_batch)

                if not batch_success:
                    success = False
                    current_batch_size = current_batch_size // 2
                    logger.info(
                        f"Reducing batch size to {current_batch_size} due to GPU memory constraints"
                    )
                    break

                all_indices.extend(indices)
                all_distances.extend(distances)
                all_test_idx.extend(test_idx)

            if success:
                break

        if current_batch_size < min_batch_size:
            logger.info(
                f"current_batch_size: {current_batch_size}, min_batch_size: {min_batch_size}"
            )
            raise RuntimeError(
                "Unable to process even with minimum batch size. Consider using CPU mode or reducing data dimensionality."
            )

        logger.info(f"Final batch size: {current_batch_size}")
        computation_time = time.time() - t0
        logger.info(f"Neighbor computation cost time: {computation_time:.3f}s")

        # Log performance metrics
        if computation_time > 0:
            logger.info(f"Processing rate: {len(test_data) / computation_time:.1f} vectors/second")

        all_indices = np.array(all_indices)
        all_distances = np.array(all_distances)

        result = np.empty(
            all_indices.shape, dtype=[(self.pk_field_name, "int64"), ("distance", "float64")]
        )
        for i in range(all_indices.shape[0]):
            for j in range(all_indices.shape[1]):
                result[i, j] = (train_idx[all_indices[i, j]], all_distances[i, j])

        df_neighbors = pd.DataFrame(
            {self.test_pk_field_name: all_test_idx, "neighbors_id": result.tolist()}
        )

        temp_manager = TempFolderManager(self.neighbors)
        temp_manager.ensure_dir(tmp_path)
        file_name = (
            f"{tmp_path}/neighbors-test-{test_batch_index}-train-{train_batch_index}.parquet"
        )
        logger.info(f"Writing neighbors to {file_name}")
        with self.neighbors.fs.open(file_name, "wb") as f:
            df_neighbors.to_parquet(f, engine="pyarrow", compression="snappy")

    def merge_neighbors(
        self, final_file_name: str | None = None, tmp_path: str | None = None
    ) -> str:
        """Merge intermediate neighbor results with separate id and distance handling."""
        t_start = time.time()
        file_list = self.neighbors.fs.glob(f"{tmp_path}/*.parquet")
        logger.info(f"Starting parallel file reading for {len(file_list)} files...")

        # Parallel file reading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    parallel_read_parquet, f, self.neighbors.fs, self.test_pk_field_name
                )
                for f in file_list
            ]
            results = list(
                tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Reading files",
                )
            )

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
        df = pd.DataFrame(
            {
                self.test_pk_field_name: test_idx,
                "neighbors_id": final_ids.tolist(),
                "neighbors_distance": final_distances.tolist(),
                "metric": self.metric_type,
                "query_expr": self.query_expr,
                "pk_field_name": self.pk_field_name,
                "test_pk_field_name": self.test_pk_field_name,
                "vector_field_name": self.vector_field_name,
                "top_k": self.top_k,
            }
        )
        logger.info(f"DataFrame creation completed in {time.time() - t_df:.3f}s")

        # Write results
        t_write = time.time()
        with self.neighbors.fs.open(final_file_name, "wb") as f:
            df.to_parquet(
                f,
                engine="pyarrow",
                compression="snappy",
                use_dictionary=False,
                row_group_size=100000,
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
        final_df = final_df.sort_values(self.test_pk_field_name).reset_index(drop=True)

        final_file_name = self.file_name
        logger.info(f"Writing final merged results to {final_file_name}")

        with self.neighbors.fs.open(final_file_name, "wb") as f:
            final_df.to_parquet(f, engine="pyarrow", compression="snappy")

        logger.info(f"Final merge completed. Total time: {time.time() - t0}")

        # Clean up partial files
        for file in partial_files:
            self.neighbors.fs.rm(file)
        logger.info("Cleaned up partial result files")

    def _get_safe_folder_name(self) -> str:
        """Generate a safe folder name based on final file naming pattern."""
        import hashlib
        import re

        # Use the same pattern as final file naming for consistency, with batch size info
        folder_name = f"neighbors-vector-{self.vector_field_name}-pk-{self.pk_field_name}-expr-{self.query_expr}-metric-{self.metric_type}-testbatch-{self.test_batch_size}-trainbatch-{self.max_rows_per_epoch}"

        # Replace only filesystem-unsafe characters, keep spaces and common operators readable
        # Replace: / \ : * ? " < > | with underscores, but keep spaces, ==, !=, etc.
        safe_folder_name = re.sub(r'[/\\:*?"<>|]', "_", folder_name)

        # If the name is too long, use a hash
        if len(safe_folder_name) > 200:
            hash_obj = hashlib.md5(folder_name.encode())
            safe_folder_name = f"neighbors_{hash_obj.hexdigest()[:12]}"

        return safe_folder_name

    def _check_resume_capability(self, tmp_path: str, test_count: int) -> tuple[list[str], bool]:
        """Check for existing results to enable resume capability."""
        existing_files = self._check_existing_results(tmp_path)
        if existing_files:
            logger.info("Resume capability: Found existing partial results")
            logger.info(
                "If you want to restart from scratch, please manually delete the temp folder"
            )
            logger.info(f"Temp folder location: {tmp_path}")

            # Skip to final merge if we already have all expected results
            if len(existing_files) >= test_count:
                logger.info("All partial results exist, proceeding to final merge")
                return existing_files, True

        return existing_files or [], False

    def _check_test_batch_status(
        self, tmp_path: str, safe_folder_name: str, i: int, train_count: int
    ) -> tuple[str, bool, bool]:
        """Check if test batch is already completed or partially completed."""
        expected_result_file = f"{tmp_path}/neighbors-{safe_folder_name}-{i}.parquet"

        # Check if final result exists
        if self.neighbors.fs.exists(expected_result_file):
            return expected_result_file, True, False

        # Check for partial train results
        test_split_path = f"{tmp_path}/tmp_{safe_folder_name}_{i}"
        partial_train_files = []
        for j in range(train_count):
            train_result_file = f"{test_split_path}/neighbors-test-{i}-train-{j}.parquet"
            if self.neighbors.fs.exists(train_result_file):
                partial_train_files.append(train_result_file)

        skip_train_computation = len(partial_train_files) == train_count

        if skip_train_computation:
            logger.info(
                f"Resuming test batch {i+1} - found all {train_count} partial train results, proceeding to merge"
            )
        elif partial_train_files:
            logger.info(
                f"Found {len(partial_train_files)}/{train_count} partial train results for test batch {i+1}"
            )
            logger.info("Will recompute missing train batches")

        return expected_result_file, False, skip_train_computation

    def _process_train_batches(
        self,
        test_data,
        train_data_generator,
        tmp_test_split_path: str,
        train_count: int,
        total_train_rows: int,
        i: int,
        skip_train_computation: bool,
        force: bool = False,
    ):
        """Process all train batches for a given test batch."""
        if not skip_train_computation:
            processed_train_rows = 0
            for j, train_train in enumerate(train_data_generator):
                # Apply query expression filtering to train data if provided
                if self.query_expr is not None:
                    original_train_size = len(train_train)
                    train_train = train_train.query(self.query_expr)
                    filtered_train_size = len(train_train)
                    logger.info(
                        f"Train batch {j+1}: Applied query_expr '{self.query_expr}' - {original_train_size} rows -> {filtered_train_size} rows ({filtered_train_size/original_train_size*100:.1f}% retained)"
                    )

                    # Skip this batch if no data remains after filtering
                    if len(train_train) == 0:
                        logger.info(f"Skipping train batch {j+1} - no data remains after filtering")
                        continue

                # Check if this specific train batch result already exists
                train_result_file = f"{tmp_test_split_path}/neighbors-test-{i}-train-{j}.parquet"
                if not force and self.neighbors.fs.exists(train_result_file):
                    logger.info(f"Skipping train batch {j+1}/{train_count} - result already exists")
                    processed_train_rows += len(train_train)
                    continue

                processed_train_rows += len(train_train)
                train_progress = (processed_train_rows / total_train_rows) * 100
                logger.info(
                    f"Computing neighbors for train batch {j+1}/{train_count} ({train_progress:.2f}% of train data)"
                )
                logger.info(f"Train batch size: {len(train_train)}")
                self.compute_neighbors(
                    test_data,
                    train_train,
                    self.vector_field_name,
                    tmp_test_split_path,
                    test_batch_index=i,
                    train_batch_index=j,
                )
        else:
            logger.info("All train batch results exist, skipping computation")

    def compute_ground_truth(self, force: bool = False):
        """Compute ground truth with resume capability.

        Args:
            force (bool): If True, force recomputation even if results already exist (default: False)
        """
        logger.info("Computing ground truth")
        if self.query_expr is not None:
            logger.info(f"Using query expression for data filtering: '{self.query_expr}'")
        else:
            logger.info("No query expression provided - processing all data")
        start_time = time.time()

        # Check if final result already exists
        if not force and self.neighbors.fs.exists(self.file_name):
            logger.info(f"Ground truth results already exist at {self.file_name}")
            logger.info("Use force=True to recompute")
            return
        elif force and self.neighbors.fs.exists(self.file_name):
            logger.info("Forcing recomputation - existing results will be overwritten")
            # Clean up existing temp folders to ensure fresh start
            self.clean_temp_folders()

        # GPU computation without caching to avoid OOM issues
        if self.use_gpu:
            logger.info("Starting ground truth computation with GPU acceleration")

        batch_info = self._initialize_batch_computation()
        temp_manager, safe_folder_name = self._setup_temp_management()

        with temp_manager.temp_folder(f"tmp_{safe_folder_name}", persistent=True) as tmp_path:
            partial_files = self._process_all_test_batches(
                tmp_path, safe_folder_name, batch_info, start_time, force
            )
            self.merge_final_results(partial_files)

        final_time = time.time() - start_time
        logger.info(f"Ground truth computation completed in {final_time:.2f}s")
        self._cleanup_temp_folders(tmp_path)

    def _initialize_batch_computation(self) -> dict:
        """Initialize batch computation parameters."""
        total_test_rows = len(self.dataset_dict["test"])
        total_train_rows = len(self.dataset_dict["train"])
        test_count = math.ceil(total_test_rows / self.test_batch_size)
        train_count = math.ceil(total_train_rows / self.max_rows_per_epoch)

        logger.info(f"Total test batches: {test_count}, total test rows: {total_test_rows}")
        logger.info(f"Total train batches: {train_count}, total train rows: {total_train_rows}")

        return {
            "total_test_rows": total_test_rows,
            "total_train_rows": total_train_rows,
            "test_count": test_count,
            "train_count": train_count,
        }

    def _setup_temp_management(self):
        """Setup temporary folder management."""
        temp_manager = TempFolderManager(self.neighbors)
        safe_folder_name = self._get_safe_folder_name()
        return temp_manager, safe_folder_name

    def _process_all_test_batches(
        self,
        tmp_path: str,
        safe_folder_name: str,
        batch_info: dict,
        start_time: float,
        force: bool = False,
    ) -> list:
        """Process all test batches and return partial files."""
        # Check for resume capability
        partial_files, can_resume = self._check_resume_capability(
            tmp_path, batch_info["test_count"]
        )
        if can_resume:
            final_time = time.time() - start_time
            logger.info(f"Ground truth computation completed (resumed) in {final_time:.2f}s")
            return partial_files

        # Process each test batch
        test_data_generator = self.dataset_dict["test"].read(
            mode="batch", batch_size=self.test_batch_size
        )
        processed_test_rows = 0

        for i, test_data in enumerate(test_data_generator):
            # Apply query expression filtering to test data if provided
            if self.query_expr is not None:
                original_test_size = len(test_data)
                test_data = test_data.query(self.query_expr)
                filtered_test_size = len(test_data)
                logger.info(
                    f"Test batch {i+1}: Applied query_expr '{self.query_expr}' - {original_test_size} rows -> {filtered_test_size} rows ({filtered_test_size/original_test_size*100:.1f}% retained)"
                )

                # Skip this batch if no data remains after filtering
                if len(test_data) == 0:
                    logger.info(f"Skipping test batch {i+1} - no data remains after filtering")
                    continue

            partial_file = self._process_single_test_batch(
                i,
                test_data,
                tmp_path,
                safe_folder_name,
                batch_info,
                processed_test_rows,
                start_time,
                force,
            )
            if partial_file:
                partial_files.append(partial_file)
            processed_test_rows += len(test_data)

        total_time = time.time() - start_time
        logger.info(f"All test batches processed in {total_time:.2f}s")
        return partial_files

    def _process_single_test_batch(
        self,
        i: int,
        test_data,
        tmp_path: str,
        safe_folder_name: str,
        batch_info: dict,
        processed_test_rows: int,
        start_time: float,
        force: bool = False,
    ) -> str:
        """Process a single test batch and return the partial file path."""
        expected_result_file, is_completed, skip_train_computation = self._check_test_batch_status(
            tmp_path, safe_folder_name, i, batch_info["train_count"]
        )

        if is_completed:
            logger.info(
                f"Skipping test batch {i+1}/{batch_info['test_count']} - final result already exists"
            )
            return expected_result_file

        # Log progress
        batch_start_time = time.time()
        processed_test_rows += len(test_data)
        progress = (processed_test_rows / batch_info["total_test_rows"]) * 100
        elapsed_time = time.time() - start_time
        eta = (
            (elapsed_time / processed_test_rows)
            * (batch_info["total_test_rows"] - processed_test_rows)
            if processed_test_rows > 0
            else 0
        )

        logger.info(
            f"Processing test batch {i+1}/{batch_info['test_count']} ({progress:.2f}% complete)"
        )
        logger.info(
            f"Test batch size: {len(test_data)}, Elapsed: {elapsed_time:.2f}s, ETA: {eta:.2f}s"
        )

        # Process this test batch
        temp_manager = TempFolderManager(self.neighbors)
        with temp_manager.temp_folder(
            f"tmp_{safe_folder_name}_{i}", persistent=True
        ) as tmp_test_split_path:
            train_data_generator = self.dataset_dict["train"].read(
                mode="batch", batch_size=self.max_rows_per_epoch
            )
            self._process_train_batches(
                test_data,
                train_data_generator,
                tmp_test_split_path,
                batch_info["train_count"],
                batch_info["total_train_rows"],
                i,
                skip_train_computation,
                force,
            )

            # Merge results for this test batch
            merged_file_name = f"{tmp_path}/neighbors-{safe_folder_name}-{i}.parquet"
            partial_file = self.merge_neighbors(merged_file_name, tmp_test_split_path)

        batch_time = time.time() - batch_start_time
        logger.info(f"Completed test batch {i+1} in {batch_time:.2f}s")
        return partial_file

    def _cleanup_temp_folders(self, base_tmp_path: str):
        """Clean up temporary folders after successful completion.

        Args:
            base_tmp_path (str): Base path containing temp folders to clean
        """
        try:
            if self.neighbors.fs.exists(base_tmp_path):
                logger.info(f"Cleaning up temporary folders: {base_tmp_path}")
                self.neighbors.fs.rm(base_tmp_path, recursive=True)
                logger.info("Temporary folders cleaned up successfully")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary folders: {e}")
            logger.warning(f"You may need to manually delete: {base_tmp_path}")

    def clean_temp_folders(self):
        """Manually clean up all temporary folders for this computation.

        This can be called to clean up persistent temp folders left from previous runs.
        """
        safe_folder_name = self._get_safe_folder_name()
        base_tmp_path = f"{self.neighbors.root_path}/{self.neighbors.name}/{self.neighbors.split}/tmp_{safe_folder_name}"

        logger.info(f"Manually cleaning temp folders: {base_tmp_path}")
        self._cleanup_temp_folders(base_tmp_path)
