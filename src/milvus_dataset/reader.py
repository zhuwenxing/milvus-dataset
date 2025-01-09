"""
Reader module for efficient dataset reading and streaming.

This module provides functionality for reading datasets with support for
both batch and streaming modes, offering flexible data access patterns
for different use cases.
"""

__all__ = ["DatasetReader"]

import json
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ml_dtypes import bfloat16
from pymilvus import DataType

from .log_config import logger
from .utils import get_fields_needs_data

if TYPE_CHECKING:
    from .core import Dataset

class DatasetReader:
    """A class for reading data from a dataset.

    This class provides methods for reading dataset contents in different modes,
    supporting both batch processing and streaming access patterns.

    Args:
        dataset (Dataset): The dataset to read from

    Examples:
        >>> reader = DatasetReader(dataset)
        >>> # Read entire dataset as DataFrame
        >>> df = reader.read(mode="batch")
        >>> # Stream data in batches
        >>> for batch in reader.read(mode="stream", batch_size=1000):
        ...     process_batch(batch)
    """

    def __init__(self, dataset: "Dataset") -> None:
        self.dataset = dataset

    def read(
        self, mode: str = "stream", batch_size: int | None = None
    ) -> pd.DataFrame | Generator[Any, Any, None] | Any:
        path = f"{self.dataset.root_path}/{self.dataset.name}/{self.dataset.split}"
        logger.info(f"Attempting to read dataset from path: {path}")

        try:
            # Check if the path exists
            if not self.dataset.fs.exists(path):
                logger.warning(f"Dataset path does not exist: path={path}")

            # List the contents of the directory for debugging
            try:
                contents = self.dataset.fs.ls(path)
                logger.debug(f"Directory contents: path={path}, files={contents}")
            except Exception as e:
                logger.warning(f"Failed to list directory contents: path={path}, error={e!s}")

            if mode == "full":
                return self._read_full(path)
            elif mode == "stream":
                return self._read_stream(path, 1)
            elif mode == "batch":
                if batch_size is None:
                    raise ValueError("Batch size must be provided when using 'batch' read mode.")
                return self._read_stream(path, batch_size)
            else:
                raise ValueError("Invalid read mode. Expected 'stream', 'batch', or 'full'.")

        except Exception as e:
            logger.exception(f"Unexpected error reading dataset: path={path}, error={e!s}")
            raise

    def process_special_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将特殊数据类型转换回原始格式
        - JSON字符串转回字典
        - FLOAT16_VECTOR从uint8转回float16数组
        - BFLOAT16_VECTOR从uint8转回bfloat16数组
        - BINARY_VECTOR从uint8转回原始二进制数组
        """
        fields_needs_data = get_fields_needs_data(self.dataset.schema)

        for field in fields_needs_data:
            if field.name not in df.columns:
                continue
            if field.dtype == DataType.JSON:
                df[field.name] = df[field.name].apply(self._convert_from_json)
            elif field.dtype == DataType.ARRAY and field.element_type == DataType.JSON:
                df[field.name] = df[field.name].apply(self._convert_from_json_array)
            elif field.dtype == DataType.FLOAT16_VECTOR:
                dim = field.params["dim"]
                df[field.name] = df[field.name].apply(lambda x: self._convert_from_float16_vector(x, dim))
            elif field.dtype == DataType.BFLOAT16_VECTOR:
                dim = field.params["dim"]
                df[field.name] = df[field.name].apply(lambda x: self._convert_from_bfloat16_vector(x, dim))
            elif field.dtype == DataType.BINARY_VECTOR:
                dim = field.params["dim"]
                df[field.name] = df[field.name].apply(lambda x: self._convert_from_binary_vector(x, dim))
            elif field.dtype == DataType.SPARSE_FLOAT_VECTOR:
                df[field.name] = df[field.name].apply(self._convert_from_json)
        return df

    def _convert_from_json(self, x):
        if x is None:
            return None
        return json.loads(x)

    def _convert_from_json_array(self, x):
        if x is None:
            return None
        return [self._convert_from_json(item) for item in x]

    def _convert_from_float16_vector(self, x):
        x = np.array(x, dtype=np.uint8)
        return x.view(np.float16)

    def _convert_from_bfloat16_vector(self, x):
        x = np.array(x, dtype=np.uint8)
        return x.view(bfloat16)

    def _convert_from_binary_vector(self, x):
        x = np.array(x, dtype=np.uint8)
        return np.unpackbits(x)

    def _read_full(self, path):
        if self.dataset.fs.isfile(path):
            with self.dataset.fs.open(path, "rb") as f:
                logger.info(f"Reading dataset: path={path}")
                df = pq.read_table(f).to_pandas()
                return self.process_special_data_types(df)
        else:
            logger.info(f"Reading full dataset from: {path}, this may take a while...")
            file_list = self.dataset.fs.glob(f"{path}/*.parquet")
            if not file_list:
                logger.warning(f"No parquet files found: path={path}")
                return pd.DataFrame()
            else:
                logger.info(f"Found {len(file_list)} parquet files.")
                dfs = []
                for file in file_list:
                    logger.debug(f"Reading file: path={file}")
                    with self.dataset.fs.open(file, "rb") as f:
                        df = pq.read_table(f).to_pandas()
                        df = self.process_special_data_types(df)
                        dfs.append(df)
                result = pd.concat(dfs, ignore_index=True)
                logger.info(f"Successfully read dataset: path={path}, rows={len(result)}")
                return result

    def _read_stream(self, path, batch_size):
        def data_generator():
            current_batch = []
            current_size = 0

            for file in self.dataset.fs.glob(f"{path}/*.parquet"):
                with self.dataset.fs.open(file, "rb") as f:
                    pf = pq.ParquetFile(f)
                    for batch in pf.iter_batches():
                        df = batch.to_pandas()
                        df = self.process_special_data_types(df)
                        current_batch.append(df)
                        current_size += len(df)

                        while current_size >= batch_size:
                            # Combine and yield a full batch
                            combined_df = pd.concat(current_batch, ignore_index=True)

                            batch_df = combined_df.iloc[:batch_size]
                            logger.info(f"Yielding batch of size {len(batch_df)}")
                            yield batch_df

                            # Keep the remainder for the next batch
                            if len(combined_df) > batch_size:
                                current_batch = [combined_df.iloc[batch_size:]]
                                current_size = len(current_batch[0])
                            else:
                                current_batch = []
                                current_size = 0

            # Yield any remaining data
            if current_batch:
                yield pd.concat(current_batch, ignore_index=True)

        return data_generator()
