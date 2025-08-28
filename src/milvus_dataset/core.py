import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from ml_dtypes import bfloat16
from pydantic import BaseModel
from pymilvus import (
    BulkInsertState,
    Collection,
    CollectionSchema,
    DataType,
    MilvusClient,
    connections,
    utility,
)

from .log_config import logger
from .neighbors import NeighborsComputation
from .reader import DatasetReader
from .storage import StorageConfig, StorageType, _create_filesystem, copy_data
from .utils import (
    ModelScopeDatasetUploader,
    gen_row_data_by_schema,
    get_bfloat16_vec_field_name_list,
    get_binary_vec_field_name_list,
    get_float16_vec_field_name_list,
    get_json_field_name_list,
    get_sparse_vec_field_name_list,
)
from .writer import DatasetWriter


class DatasetConfig(BaseModel):
    storage: StorageConfig
    default_schema: dict[str, Any] | None = None


class ConfigManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.config = None
        return cls._instance

    def get_config(self) -> DatasetConfig:
        if self.config is None:
            raise ValueError("ConfigManager has not been initialized")
        return self.config

    def init_storage(
        self, root_path: str, storage_type: StorageType = StorageType.LOCAL, options=None
    ):
        logger.info("Initializing storage")
        if storage_type == StorageType.S3:
            options = self._prepare_s3_options(options)
            self._verify_s3_connection(root_path, options)
        config = DatasetConfig(
            storage=StorageConfig(storage_type=storage_type, root_path=root_path, options=options)
        )
        self._initialize(config)

    def _prepare_s3_options(self, options: dict[str, Any]) -> dict[str, Any]:
        s3_options = {
            "key": options.get("aws_access_key_id")
            or options.get("access_key")
            or options.get("key"),
            "secret": options.get("aws_secret_access_key")
            or options.get("secret_key")
            or options.get("secret"),
            "client_kwargs": {},
        }

        if "endpoint_url" in options:
            s3_options["client_kwargs"]["endpoint_url"] = options["endpoint_url"]
        if "region_name" in options:
            s3_options["client_kwargs"]["region_name"] = options["region_name"]
        if "use_ssl" in options:
            s3_options["use_ssl"] = options["use_ssl"]

        return s3_options

    def _verify_s3_connection(self, root_path: str, options: dict[str, Any]):
        try:
            logger.info("Connecting to S3/MinIO")
            fs = fsspec.filesystem("s3", **options)

            try:
                # try to create a small file to verify the connection
                temp_file_path = os.path.join(root_path, f"milvus-dataset-{int(time.time())}.txt")
                with fs.open(temp_file_path, "w") as f:
                    f.write("test")
                fs.rm(temp_file_path)
                logger.info(f"Root path '{root_path}' is accessible")
            except Exception as e:
                logger.error(
                    f"Failed to access root path '{root_path}' on S3/MinIO: {e!s}, "
                    "please check your access key, secret key, endpoint_url, and region_name"
                )
                raise

        except Exception as e:
            logger.error(
                f"Failed to connect to S3/MinIO: {e!s}, please check your access key, "
                "secret key, endpoint_url, and region_name"
            )
            raise

    def _initialize(self, config: DatasetConfig):
        with self._lock:
            self.config = config


def get_config() -> DatasetConfig:
    return ConfigManager().get_config()


class Dataset:
    def __init__(self, name: str, split="train"):
        self.name = name
        self.config = get_config()
        self.storage = self.config.storage
        self.fs = _create_filesystem(self.storage)
        self.root_path = self.storage.root_path
        self._schema = None
        self.metadata = self._load_metadata()
        self.split = split
        logger.info(f"Initializing dataset '{name}' with split '{split}'")
        self._ensure_split_exists()
        self.writer = DatasetWriter(self)
        self.reader = DatasetReader(self)
        self._summary = None

    def __repr__(self):
        summary = self._get_summary()
        return (
            f"Dataset(name='{summary['name']}', split='{summary['split']}', "
            f"num_rows={summary['num_rows']}, "
            f"num_columns={summary['num_columns']}, "
            f"size={summary['size']} MB, "
            f"num_files={summary['num_files']})"
        )

    def __len__(self) -> int:
        """Get the total number of rows in the dataset.

        Returns:
            int: Total number of rows
        """
        total_rows = 0
        for file_path in self.fs.glob(f"{self.root_path}/{self.name}/{self.split}/*.parquet"):
            with self.fs.open(file_path, "rb") as f:
                metadata = pq.read_metadata(f)
                total_rows += metadata.num_rows
        return total_rows

    def set_schema(self, schema: CollectionSchema):
        """Set the schema for the dataset."""
        self._schema = schema
        self._save_schema()
        logger.info(f"Schema set for dataset '{self.name}'")

    def _save_schema(self):
        """Save the schema to a file."""
        schema_path = f"{self.root_path}/{self.name}/schema.json"
        with self.fs.open(schema_path, "w") as f:
            json.dump(self._schema.to_dict(), f, indent=2)

    def _load_schema(self):
        """Load the schema from a file if it exists."""
        schema_path = f"{self.root_path}/{self.name}/schema.json"
        if self.fs.exists(schema_path):
            with self.fs.open(schema_path, "r") as f:
                schema_dict = json.load(f)
            return CollectionSchema.construct_from_dict(schema_dict)
        return None

    def _ensure_split_exists(self) -> None:
        """Ensure the split directory exists.

        Creates the directory for the current split if it doesn't exist.
        """
        split_dir = f"{self.root_path}/{self.name}/{self.split}"
        self.fs.makedirs(split_dir, exist_ok=True)

    def _load_metadata(self):
        metadata_path = f"{self.root_path}/{self.name}/metadata.json"
        if self.fs.exists(metadata_path):
            with self.fs.open(metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        metadata_path = f"{self.root_path}/{self.name}/metadata.json"
        with self.fs.open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def get_schema(self) -> CollectionSchema | None:
        """Get the current schema of the dataset."""
        if self._schema is None:
            self._schema = self._load_schema()
        return self._schema

    def create_schema_model(self):  # noqa: C901
        """Create a PyArrow schema to validate the dataset's schema."""
        fields = []
        for schema in self._schema.fields:
            if schema.dtype == DataType.VARCHAR:
                pa_type = pa.string()
            elif schema.dtype == DataType.STRING:
                pa_type = pa.string()
            elif schema.dtype == DataType.BOOL:
                pa_type = pa.bool_()
            elif schema.dtype == DataType.INT8:
                pa_type = pa.int8()
            elif schema.dtype == DataType.INT16:
                pa_type = pa.int16()
            elif schema.dtype == DataType.INT32:
                pa_type = pa.int32()
            elif schema.dtype == DataType.INT64:
                pa_type = pa.int64()
            elif schema.dtype == DataType.FLOAT:
                pa_type = pa.float32()
            elif schema.dtype == DataType.DOUBLE:
                pa_type = pa.float64()
            elif schema.dtype == DataType.FLOAT_VECTOR:
                pa_type = pa.list_(pa.float32(), schema.dim)
            elif schema.dtype == DataType.FLOAT16_VECTOR:
                pa_type = pa.list_(pa.uint8(), schema.dim * 2)
            elif schema.dtype == DataType.BFLOAT16_VECTOR:
                pa_type = pa.list_(pa.uint8(), schema.dim * 2)
            elif schema.dtype == DataType.BINARY_VECTOR:
                pa_type = pa.list_(pa.uint8(), schema.dim // 8)
            elif schema.dtype == DataType.SPARSE_FLOAT_VECTOR:
                # 使用struct类型存储稀疏向量
                pa_type = pa.string()
            elif schema.dtype == DataType.ARRAY:
                element_type = schema.element_type
                if element_type == DataType.BOOL:
                    pa_type = pa.list_(pa.bool_())
                elif element_type == DataType.INT8:
                    pa_type = pa.list_(pa.int8())
                elif element_type == DataType.INT16:
                    pa_type = pa.list_(pa.int16())
                elif element_type == DataType.INT32:
                    pa_type = pa.list_(pa.int32())
                elif element_type == DataType.INT64:
                    pa_type = pa.list_(pa.int64())
                elif element_type == DataType.FLOAT:
                    pa_type = pa.list_(pa.float32())
                elif element_type == DataType.DOUBLE:
                    pa_type = pa.list_(pa.float64())
                elif element_type == DataType.VARCHAR:
                    pa_type = pa.list_(pa.string())
                else:
                    raise ValueError(f"Unsupported array element type: {element_type}")
            elif schema.dtype == DataType.JSON:
                pa_type = pa.string()  # JSON objects are stored as strings
            else:
                raise ValueError(f"Unsupported data type: {schema.dtype}")
            fields.append(pa.field(schema.name, pa_type))

        arrow_schema = pa.schema(fields)
        return arrow_schema

    def format_data(self, data):  # noqa
        schema = self._schema
        binary_vector_field_names = get_binary_vec_field_name_list(schema)
        sparse_vector_field_names = get_sparse_vec_field_name_list(schema)
        float16_vector_field_names = get_float16_vec_field_name_list(schema)
        bfloat16_vector_field_names = get_bfloat16_vec_field_name_list(schema)
        json_fields = get_json_field_name_list(schema)
        for field in binary_vector_field_names:

            def covert_binary(x):
                if isinstance(x, np.ndarray) and x.dtype == np.uint8:
                    return x
                else:
                    return np.array(np.packbits(x, axis=-1), dtype=np.uint8)

            data[field] = data[field].apply(lambda x: covert_binary(x))
        for field in sparse_vector_field_names:

            def convert_sparse(x):
                if isinstance(x, str):
                    return x
                elif isinstance(x, dict) and "indices" in x and "values" in x:
                    x = dict(zip(x["indices"], x["values"], strict=False))
                    return json.dumps(x)
                elif isinstance(x, dict) and not ("indices" in x and "values" in x):
                    return json.dumps(x)
                else:
                    raise ValueError(
                        f"Unsupported sparse vector format {type(x)}, only coo and dok format supported"
                    )

            data[field] = data[field].apply(lambda x: convert_sparse(x))
        for field in float16_vector_field_names:

            def convert_float16(x):
                if isinstance(x, np.ndarray) and x.dtype == np.uint8:
                    return x
                else:
                    return np.array(
                        np.array(x, dtype=np.float16).view(np.uint8).tolist(),
                        dtype=np.dtype("uint8"),
                    )

            data[field] = data[field].apply(lambda x: convert_float16(x))
        for field in bfloat16_vector_field_names:

            def convert_bfloat16(x):
                if isinstance(x, np.ndarray) and x.dtype == np.uint8:
                    return x
                else:
                    return np.array(
                        np.array(x, dtype=bfloat16).view(np.uint8).tolist(),
                        dtype=np.dtype("uint8"),
                    )

            data[field] = data[field].apply(lambda x: convert_bfloat16(x))
        for field in json_fields:

            def convert_json(x):
                if isinstance(x, dict):
                    return json.dumps(x)
                elif isinstance(x, list):
                    return [json.dumps(d) for d in x]
                elif isinstance(x, str):
                    return x
                else:
                    raise ValueError(f"Unsupported json field format: {type(x)}")

            data[field] = data[field].apply(lambda x: convert_json(x))

        return data

    def verify_schema(self, values):
        """Validate data using PyArrow schema"""
        if isinstance(values, pd.DataFrame):
            try:
                # format the binary/f16/bf16/sparse vector, json data
                # Validate basic types
                values = self.format_data(values)
                table = pa.Table.from_pandas(values, schema=self.create_schema_model())
                logger.info(f"Data validation passed {table}")
                df = table.to_pandas()
                logger.info(f"Data validation passed \n{df}")
                return df
            except pa.ArrowInvalid as e:
                raise ValueError(f"Data validation failed: {e!s}") from e
        elif isinstance(values, dict | list):
            try:
                df = pd.DataFrame(values)
                return self.verify_schema(df)
            except (pa.ArrowInvalid, ValueError) as e:
                raise ValueError(f"Data validation failed: {e!s}") from e
        else:
            raise ValueError(f"Unsupported data type: {type(values)}")

    def _get_summary(self) -> dict[str, str | int | dict]:
        # TODO: add a function to sort the files in the format of train/test-{index:5d}-of-{total:5d}.parquet
        if self._summary is None:
            path = f"{self.root_path}/{self.name}/{self.split}"
            if not self.fs.exists(path):
                self._summary = {
                    "name": self.name,
                    "split": self.split,
                    "size": 0,
                    "num_rows": 0,
                    "num_columns": 0,
                    "schema": {},
                    "num_files": 0,
                }
            else:
                total_rows = 0
                total_size = 0
                schema_dict = {}
                num_files = 0

                if self.split in ["train", "test"]:
                    # Sort files by creation time
                    files = []
                    for f in self.fs.glob(f"{path}/*.parquet"):
                        info = self.fs.info(f)
                        # Try different timestamp fields based on storage type
                        timestamp = (
                            info.get("created") or info.get("LastModified") or info.get("mtime", 0)
                        )
                        files.append((f, timestamp))
                    files.sort(key=lambda x: x[1])  # Sort by creation time

                    # Rename files according to their sorted order
                    for idx, (old_file, _) in enumerate(files, 1):
                        new_name = f"{path}/{self.split}-{idx:05d}-of-{len(files):05d}.parquet"
                        if old_file != new_name:
                            self.fs.rename(old_file, new_name)
                    files = [
                        f"{path}/{self.split}-{idx:05d}-of-{len(files):05d}.parquet"
                        for idx in range(1, len(files) + 1)
                    ]
                else:
                    files = self.fs.glob(f"{path}/*.parquet")

                logger.info(f"files in path {path}: {files}")
                for file in files:
                    num_files += 1
                    try:
                        with self.fs.open(file, "rb") as f:
                            parquet_file = pq.ParquetFile(f)
                            total_rows += parquet_file.metadata.num_rows
                            if not schema_dict:
                                schema = parquet_file.schema.to_arrow_schema()
                                schema_dict = {field.name: str(field.type) for field in schema}
                            total_size += self.fs.info(file)["size"]
                    except Exception as e:
                        logger.error(f"Error reading file {file}: {e!s}")
                        continue

                self._summary = {
                    "name": self.name,
                    "split": self.split,
                    "size": f"{total_size / 1024 / 1024:.3f} MB",
                    "num_rows": total_rows,
                    "num_columns": len(schema_dict),
                    "schema": schema_dict,
                    "num_files": num_files,
                }

        return self._summary

    def get_features(self) -> dict[str, str]:
        """
        Get the features (schema) of the dataset.

        Returns:
            Dict[str, str]: A dictionary where the keys are field names and the values are the data types as strings.
        """
        summary = self._get_summary()
        return summary["schema"]

    def get_num_rows(self) -> int:
        """
        Get the number of rows in the dataset.

        Returns:
            int: The total number of rows in the dataset.
        """
        summary = self._get_summary()
        return summary["num_rows"]

    def get_num_columns(self) -> int:
        """
        Get the number of columns in the dataset.

        Returns:
            int: The number of columns in the dataset.
        """
        summary = self._get_summary()
        return summary["num_columns"]

    def get_size(self) -> int:
        """
        Get the size of the dataset in bytes.

        Returns:
            int: The total size of the dataset in bytes.
        """
        summary = self._get_summary()
        return summary["size"]

    def get_num_files(self) -> int:
        """
        Get the number of files in the dataset.

        Returns:
            int: The number of files in the dataset.
        """
        summary = self._get_summary()
        return summary["num_files"]

    def _prepare_for_write(self, mode: str):
        logger.info(f"Preparing to write data to dataset '{self.name}'")
        if mode == "overwrite":
            files = self.fs.glob(f"{self.root_path}/{self.name}/{self.split}/*.parquet")
            logger.info(f"Deleting existing dataset '{self.name}' split '{self.split}': {files}")
            if self.fs.exists(f"{self.root_path}/{self.name}/{self.split}"):
                self.fs.rm(f"{self.root_path}/{self.name}/{self.split}", recursive=True)
            self._ensure_split_exists()

        if self._schema is None:
            self._schema = self._load_schema()
        if self._schema is None:
            raise ValueError(
                "Schema must be set before writing data. Please use set_schema() method."
            )
        if self.split not in ["train", "test"]:
            raise ValueError("Only 'train' and 'test' splits are allowed.")
        self._summary = None

    def get_writer(self, mode: str = "append", **writer_options):
        self._prepare_for_write(mode)
        return DatasetWriter(self, **writer_options)

    # def write(
    #     self,
    #     data: Union[pd.DataFrame, Dict, List[Dict]],
    #     mode: str = "append",
    #     verify_schema: bool = True,
    # ):
    #     """Write data to the dataset"""
    #     # Get the writer with the specified mode
    #     writer = self.get_writer(mode=mode)
    #     result = writer.write(data, verify_schema=verify_schema)

    #     self._summary = None
    #     return result

    def read(self, mode: str = "full", batch_size: int = 1000):
        return self.reader.read(mode, batch_size)

    def get_total_rows(self, split: str) -> int:
        # Implement this method to return the total number of rows for a given split
        pass

    def summary(self) -> dict[str, str | int | dict]:
        path = f"{self.root_path}/{self.name}/{self.split}"
        if not self.fs.exists(path):
            return {
                "name": self.name,
                "size": 0,
                "num_rows": 0,
                "num_columns": 0,
                "schema": {},
                "num_files": 0,
            }

        total_rows = 0
        total_size = 0
        schema_dict = {}
        num_files = 0

        files = self.fs.glob(f"{path}/*.parquet")

        logger.info(f"files in path: {files}")
        for file in files:
            logger.info(f"files in path: {file}")
            num_files += 1
            try:
                with self.fs.open(file, "rb") as f:
                    parquet_file = pq.ParquetFile(f)
                    total_rows += parquet_file.metadata.num_rows
                    if not schema_dict:
                        schema = parquet_file.schema.to_arrow_schema()
                        schema_dict = {field.name: str(field.type) for field in schema}
                    total_size += self.fs.info(file)["size"]
            except Exception as e:
                logger.error(f"Error reading file {file}: {e!s}")
                continue

        return {
            "name": self.name,
            "size": f"{total_size / 1024 / 1024:.3f} MB",
            "num_rows": total_rows,
            "num_columns": len(schema_dict),
            "schema": schema_dict,
            "num_files": num_files,
        }


class DatasetMetadata(BaseModel):
    name: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    train: int | None = None
    test: int | None = None
    source: str | None = None
    task: str | None = None
    dense_model: dict[str, Any] | None = None
    sparse_model: dict[str, Any] | None = None
    license: str = (
        "DISCLAIMER AND LICENSE NOTICE:\n"
        "1. This dataset is intended for benchmarking and research purposes only.\n"
        "2. The source data used in this dataset retains its original license and copyright. "
        "Users must comply with the respective licenses of the original data sources.\n"
        "3. The ground truth part of the dataset (including but not limited to annotations, "
        "labels, and evaluation metrics) is licensed under Apache 2.0.\n"
        "4. This dataset is provided 'AS IS' without any warranty. The dataset maintainers "
        "are not responsible for any copyright violations arising from the use of the source data.\n"
        "5. If you are the copyright holder of any source data and believe it has been included "
        "inappropriately, please contact us for prompt removal.\n"
        "6. Commercial use of this dataset must ensure compliance with the original data sources' "
        "licenses and obtain necessary permissions where required."
    )

    @classmethod
    def from_dataset_dict(cls, dataset_dict: "DatasetDict"):
        """Create metadata from a DatasetDict instance"""
        metadata = cls(
            name=dataset_dict.name,
            train=dataset_dict.datasets["train"].get_num_rows()
            if "train" in dataset_dict.datasets
            else None,
            test=dataset_dict.datasets["test"].get_num_rows()
            if "test" in dataset_dict.datasets
            else None,
        )
        return metadata


class DatasetDict(dict):
    def __init__(self, datasets: dict[str, Dataset]):
        super().__init__(datasets)
        self.datasets = datasets
        self.name = datasets["train"].name
        self.storage = datasets["train"].config.storage
        self._summary = {}
        self._metadata = {}
        self.meta = None
        self._load_metadata()

    def __getitem__(self, key: str) -> Dataset:
        return super().__getitem__(key)

    def to_storage(self, destination: StorageConfig):
        """
        Save the dataset by copying all files to a specified destination using the storage utilities.
        The destination can be any supported storage type (local, S3, GCS).

        Args:
            destination (StorageConfig): The storage configuration for the destination where the dataset should be saved.

        Raises:
            Exception: If there is an error copying the dataset directory
        """

        # Get source dataset directory and storage config from train split
        source_storage = self.datasets["train"].storage
        source_path = os.path.join(source_storage.root_path, self.name)
        dest_path = os.path.join(destination.root_path, self.name)

        try:
            logger.info(f"Copying dataset from {source_path} to {dest_path}")
            results = copy_data(
                source_config=source_storage,
                dest_config=destination,
                source_path=source_path,
                dest_path=dest_path,
            )

            for src, dst, status in results:
                if status == "Success":
                    logger.debug(f"Successfully copied {src} to {dst}")
                else:
                    logger.warning(f"Issue copying {src} to {dst}: {status}")

            logger.info(
                f"Dataset '{self.name}' has been successfully saved to {destination.root_path}"
            )
        except Exception as e:
            error_msg = f"Failed to copy dataset directory: {e!s}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def get_metadata(self):
        self._load_metadata()
        return self.meta

    def set_metadata(self, metadata: DatasetMetadata | dict[str, Any]):
        """Set metadata for the dataset and save it to metadata.json

        Args:
            metadata: Either a DatasetMetadata object or a dictionary containing metadata fields to update
        """
        if isinstance(metadata, dict):
            # If current metadata doesn't exist, create a new one
            if self.meta is None:
                current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
                self.meta = DatasetMetadata(created_at=current_time, updated_at=current_time)

            # Update only the provided fields
            for key, value in metadata.items():
                if hasattr(self.meta, key) and key not in ["created_at", "updated_at"]:
                    setattr(self.meta, key, value)
        else:
            # Preserve existing timestamps if they exist
            if self.meta:
                metadata.created_at = self.meta.created_at
                metadata.updated_at = self.meta.updated_at
            self.meta = metadata
        logger.info(f"Updated metadata: {self.meta}")

        self._save_metadata()

    def _load_metadata(self):
        """Load metadata from metadata.json if it exists"""
        try:
            file_path = f"{self.storage.root_path}/{self.name}/metadata.json"
            with self.datasets["train"].fs.open(file_path, "r") as f:
                metadata_dict = json.load(f)
                self.meta = DatasetMetadata(**metadata_dict)
        except Exception:
            # Only initialize timestamps when metadata.json doesn't exist
            current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
            self.meta = DatasetMetadata.from_dataset_dict(self)
            self.meta.created_at = current_time
            self.meta.updated_at = current_time
            self._save_metadata()

    def _save_metadata(self):
        """Save metadata to metadata.json"""
        if self.meta is None:
            return

        metadata_file = f"{self.storage.root_path}/{self.name}/metadata.json"
        try:
            with self.datasets["train"].fs.open(metadata_file, "w") as f:
                json.dump(self.meta.model_dump(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def to_readme(self):
        # Create a readme file
        readme = f"""---
configs:
  - config_name: train
    data_files:
      - split: train
        path: train/*
  - config_name: test
    data_files:
      - split: test
        path: test/*
  - config_name: neighbors
    data_files:
      - split: neighbors
        path: neighbors/*
---

# Dataset Overview

dataset: {self.name}

## Metadata

"""
        metadata = self.get_metadata()
        if metadata:
            readme += f"""
- **Creation Time**: {metadata.created_at if metadata.created_at else 'N/A'}
- **Update Time**: {metadata.updated_at if metadata.updated_at else 'N/A'}
- **Source**: {metadata.source if metadata.source else 'N/A'}
- **Task**: {metadata.task if metadata.task else 'N/A'}
- **Train Samples**: {metadata.train if metadata.train else 'N/A'}
- **Test Samples**: {metadata.test if metadata.test else 'N/A'}
"""
            if metadata.dense_model:
                readme += f"- **Dense Model**: {json.dumps(metadata.dense_model, indent=2)}\n"
            if metadata.sparse_model:
                readme += f"- **Sparse Model**: {json.dumps(metadata.sparse_model, indent=2)}\n"
            if metadata.license:
                readme += f"- **License**: {metadata.license}\n"

        readme += "\n## Dataset Statistics\n\n"

        headers = [
            "Split",
            "Name",
            "Size",
            "Num Rows",
            "Num Columns",
            "Schema",
            "Num Files",
        ]
        table = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        logger.info(f"summary: {self._summary}")
        for split, details in self._summary.items():
            row = [
                str(split),
                str(details.get("name", "")),
                str(details.get("size", "")),
                str(details.get("num_rows", "")),
                str(details.get("num_columns", "")),
                json.dumps(details.get("schema", {}), indent=2).replace("\n", "<br>"),
                str(details.get("num_files", "")),
            ]

            logger.info(row)
            table.append("| " + " | ".join(row) + " |")
        file_path = f"{self.storage.root_path}/{self.name}/README.md"
        with self.datasets["train"].fs.open(file_path, "w") as f:
            f.write(readme + "\n".join(table) + "\n")

    def summary(self) -> dict:
        """
        Get a summary of the entire dataset dictionary.

        Returns:
            Dict: A dictionary containing the summary information for all splits.
        """
        res = {split: dataset.summary() for split, dataset in self.datasets.items()}
        # save to json file
        file_path = f"{self.storage.root_path}/{self.name}/summary.json"
        with self.datasets["train"].fs.open(file_path, "w") as f:
            json.dump(res, f, indent=2)
        self._summary = res
        self.to_readme()
        return res

    def __repr__(self):
        dataset_dict = self.summary()

        # Create the final dictionary structure
        final_dict = {"DatasetDict": dataset_dict}

        # Use json.dumps for formatting, with indentation set to 2 spaces
        return json.dumps(final_dict, indent=2)

    def to_dict(self):
        """Return the original dictionary representation of the dataset dictionary"""
        return self.summary()

    def compute_neighbors(
        self,
        vector_field_name,
        pk_field_name="id",
        test_pk_field_name=None,
        query_expr=None,
        top_k=1000,
        **kwargs,
    ):
        neighbors_computation = NeighborsComputation(
            self,
            vector_field_name,
            pk_field_name=pk_field_name,
            test_pk_field_name=test_pk_field_name,
            query_expr=query_expr,
            top_k=top_k,
            **kwargs,
        )
        neighbors_computation.compute_ground_truth()

    def get_neighbors(
        self, vector_field_name, pk_field_name="id", query_expr=None, metric_type="cosine"
    ):
        neighbors = self["neighbors"]
        file_name = f"{neighbors.root_path}/{neighbors.name}/{neighbors.split}/neighbors-vector-{vector_field_name}-pk-{pk_field_name}-expr-{query_expr}-metric-{metric_type}.parquet"
        if neighbors.fs.exists(file_name):
            with neighbors.fs.open(file_name, "rb") as f:
                return pq.read_table(f).to_pandas()
        else:
            logger.warning(f"Neighbors file not found: {file_name}")
            return pd.DataFrame()

    def set_schema(self, schema: CollectionSchema):
        """Set the schema for all splits."""
        for dataset in self.values():
            dataset.set_schema(schema)
        logger.info(f"Schema set for dataset '{self.name}' and all its splits")

    def to_milvus(
        self, milvus_config: dict, collection_name=None, mode="import", milvus_storage=None
    ) -> None:
        """
        Write the dataset to Milvus and verify search recall accuracy.
        Can be either 'insert' or 'bulk import'.
        Requires the Milvus connection information, which can be passed as a Milvus client.

        Args:
            milvus_config (Dict): The Milvus connection configuration.
            collection_name (str, optional): Name of the Milvus collection. Defaults to dataset name.
            mode (str, optional): The mode of writing to Milvus. Defaults to 'insert'.
            milvus_storage (optional): The Milvus storage configuration. Defaults to None.

        Returns:
            None
        """
        # Create collection
        milvus_client = MilvusClient(**milvus_config)
        connections.connect(**milvus_config)
        if collection_name is None:
            collection_name = self.name
        if milvus_client.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' already exists in Milvus, drop it first")
            milvus_client.drop_collection(collection_name)

        milvus_client.create_collection(
            collection_name=collection_name,
            schema=self["train"].get_schema(),
        )
        # Sync data to Milvus storage
        self.to_storage(milvus_storage)
        # List all files in train split
        # Create fs by Milvus storage
        _fs = _create_filesystem(milvus_storage)
        train_fs = self.datasets["train"].fs
        train_files = train_fs.glob(
            f"{self.datasets['train'].root_path}/{self.name}/train/*.parquet"
        )
        file_names = [file.split("/")[-1] for file in train_files]
        bulk_insert_files = [
            f"{milvus_storage.root_path}/{self.name}/train/{file}" for file in file_names
        ]

        # TODO: Restful API to import data
        task_ids = []
        for file in bulk_insert_files:
            file = "/".join(file.split("/")[1:])
            logger.info(f"Importing file {file} to Milvus")
            task_id = utility.do_bulk_insert(
                collection_name=collection_name,
                files=[file],
            )
            task_ids.append(task_id)
            logger.info(f"Create a bulk inert task, task id: {task_id}")
        # List all import task and wait complete
        while len(task_ids) > 0:
            logger.info("Wait 1 second to check bulk insert tasks state...")
            time.sleep(1)
            for id in task_ids:
                state = utility.get_bulk_insert_state(task_id=id)
                if (
                    state.state == BulkInsertState.ImportFailed
                    or state.state == BulkInsertState.ImportFailedAndCleaned
                ):
                    logger.error(f"The task {state.task_id} failed, reason: {state.failed_reason}")
                    raise ValueError(
                        f"The task {state.task_id} failed, reason: {state.failed_reason}"
                    )
                elif state.state == BulkInsertState.ImportCompleted:
                    logger.info(f"The task {state.task_id} completed with state {state}")
                    task_ids.remove(id)

        logger.info(
            f"Dataset '{self.name}' has been successfully written to Milvus collection '{collection_name}'"
        )

    def benchmark_milvus(  # noqa: C901
        self,
        collection_name: str,
        search_params: dict | None = None,
        rounds: int = 3,
        top_k: int | None = None,
        min_concurrent: int = 1,
        max_concurrent: int = 32,
        concurrent_step: int = 2,
        test_duration: int = 60,  # Duration in seconds for each test
    ) -> pd.DataFrame:
        """
        Benchmark Milvus search performance using different concurrency levels.
        Measures QPS, latency, and recall across different concurrent search configurations.

        Args:
            collection_name (str): Name of the Milvus collection to benchmark
            search_params (dict): Optional search parameters to override defaults
            rounds (int): Number of search rounds for averaging results
            top_k (int): Number of nearest neighbors to retrieve, defaults to ground truth k
            min_concurrent (int): Minimum number of concurrent processes
            max_concurrent (int): Maximum number of concurrent processes
            concurrent_step (int): Multiplier for increasing concurrency (e.g., 2 means 1,2,4,8,...)
            test_duration (int): Duration in seconds for each concurrent test

        Returns:
            pd.DataFrame: Benchmark results including QPS, latency, and recall metrics
        """
        import queue
        import threading
        from concurrent.futures import ProcessPoolExecutor
        from time import perf_counter, sleep

        from tabulate import tabulate

        logger.info("Starting Milvus benchmark...")
        collection = Collection(collection_name)

        # Create index and load collection if not already done
        if not collection.has_index():
            logger.info("Creating index...")
            metric_type = self.get_neighbors("emb")["metric"].iloc[0].upper()
            index_params = {"metric_type": metric_type, "index_type": "FLAT", "params": {}}
            collection.create_index(field_name="emb", index_params=index_params)

        if not collection.is_loaded():
            logger.info("Loading collection...")
            collection.load()

        # Prepare search data
        test_data = self["test"].read(mode="full")
        test_vectors = [np.array(v, dtype=np.float32) for v in test_data["emb"]]
        neighbors_data = self.get_neighbors("emb")
        metric_type = neighbors_data["metric"].iloc[0].upper()

        if top_k is None:
            top_k = len(neighbors_data["neighbors_id"].iloc[0])

        # Default search parameters
        default_search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
        if search_params:
            default_search_params.update(search_params)

        def search_worker(vectors, params, ready_event, start_event, stop_event, result_queue):
            """Worker function that waits for synchronization before starting the actual test"""
            try:
                # Signal that this worker is ready
                ready_event.set()
                # Wait for the start signal
                start_event.wait()

                search_count = 0
                results = []

                # Continue searching until stop event is set
                while not stop_event.is_set():
                    for chunk in vectors:
                        if stop_event.is_set():
                            break
                        search_result = collection.search(
                            data=[
                                chunk
                            ],  # Search one vector at a time for more uniform distribution
                            anns_field="emb",
                            param=params,
                            limit=top_k,
                            output_fields=["idx"],
                        )
                        results.extend(search_result)
                        search_count += 1

                # Put results in queue
                result_queue.put((search_count, results))

            except Exception as e:
                logger.error(f"Search worker error: {e}")
                result_queue.put((0, []))

        def calculate_recall(search_results, query_indices):
            recall_sum = 0
            valid_results = 0
            for hits, query_idx in zip(search_results, query_indices, strict=False):
                if hits is None:
                    continue
                gt_row = neighbors_data[neighbors_data["idx"] == query_idx]
                gt_neighbors = set(gt_row.iloc[0]["neighbors_id"])
                milvus_neighbors = {hit.entity.get("idx") for hit in hits}
                recall = len(gt_neighbors.intersection(milvus_neighbors)) / len(gt_neighbors)
                recall_sum += recall
                valid_results += 1
            return recall_sum / valid_results if valid_results > 0 else 0

        # Generate concurrency levels
        concurrent_levels = []
        current = min_concurrent
        while current <= max_concurrent:
            concurrent_levels.append(current)
            current *= concurrent_step

        # Run benchmark
        results = []
        for num_workers in concurrent_levels:
            logger.info(f"\nTesting with {num_workers} concurrent processes...")

            # Split test vectors for parallel processing
            chunk_size = max(1, len(test_vectors) // num_workers)
            vector_chunks = [
                list(test_vectors[i : i + chunk_size])  # Convert each vector to a list
                for i in range(0, len(test_vectors), chunk_size)
            ]

            round_metrics = []
            for round in range(rounds):
                logger.info(f"Round {round + 1}/{rounds}")

                # Create synchronization events and result queue
                ready_events = [threading.Event() for _ in range(num_workers)]
                start_event = threading.Event()
                stop_event = threading.Event()
                result_queue = queue.Queue()

                # Start workers
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    for i, chunk in enumerate(vector_chunks):
                        executor.submit(
                            search_worker,
                            chunk,
                            default_search_params,
                            ready_events[i],
                            start_event,
                            stop_event,
                            result_queue,
                        )
                    # Wait for all workers to be ready
                    logger.info("Waiting for workers to be ready...")
                    for event in ready_events:
                        event.wait()

                    # Start the test
                    logger.info(f"Starting {test_duration}s test...")
                    start_time = perf_counter()
                    start_event.set()

                    # Run for specified duration
                    sleep(test_duration)

                    # Stop the test
                    stop_event.set()
                    end_time = perf_counter()

                    # Collect results
                    total_searches = 0
                    all_results = []
                    while not result_queue.empty():
                        count, results = result_queue.get()
                        total_searches += count
                        all_results.extend(results)

                    # Calculate metrics
                    duration = end_time - start_time
                    qps = total_searches / duration
                    avg_latency = duration * 1000 / total_searches if total_searches > 0 else 0
                    recall = calculate_recall(all_results, test_data["idx"])

                    round_metrics.append(
                        {
                            "Concurrent": num_workers,
                            "QPS": qps,
                            "Latency(ms)": avg_latency,
                            "Recall": recall,
                        }
                    )

                    logger.info(
                        f"Round results - QPS: {qps:.2f}, Latency: {avg_latency:.2f}ms, Recall: {recall:.4f}"
                    )

            # Average metrics across rounds
            avg_metrics = {
                "Concurrent": num_workers,
                "QPS": sum(r["QPS"] for r in round_metrics) / rounds,
                "Latency(ms)": sum(r["Latency(ms)"] for r in round_metrics) / rounds,
                "Recall": sum(r["Recall"] for r in round_metrics) / rounds,
            }
            results.append(avg_metrics)

            logger.info(
                f"Average metrics - QPS: {avg_metrics['QPS']:.2f}, "
                f"Latency: {avg_metrics['Latency(ms)']:.2f}ms, "
                f"Recall: {avg_metrics['Recall']:.4f}"
            )

        # Create DataFrame with results
        df_results = pd.DataFrame(results)

        # Print formatted table
        table = tabulate(df_results, headers="keys", tablefmt="grid", floatfmt=".2f")
        logger.info(f"\nMilvus Benchmark Results:\n{table}")

        return df_results

    def to_hf(
        self,
        repo_name: str | None = None,
    ):
        """
        Upload a dataset to the Hugging Face Hub

        Args:
            repo_name: Repository name on Hugging Face Hub (format: 'username/dataset-name')
        """
        # Initialize API
        local_path = f"{self.storage.root_path}/{self.name}"
        # if storage is not local, download to local
        if self.storage.storage_type != StorageType.LOCAL:
            logger.info("Downloading dataset to local storage")
            self.to_storage(StorageConfig(storage_type=StorageType.LOCAL, root_path=local_path))
        api = HfApi()
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "Please provide a Hugging Face token or set the HF_TOKEN environment variable"
            )

        if repo_name is None:
            raise ValueError("Please provide a repository name (format: 'username/dataset-name')")

        try:
            # Ensure local path exists
            local_path = Path(local_path)
            if not local_path.exists():
                raise ValueError(f"Local path does not exist: {local_path}")

            # Create or get repository
            repo_url = api.create_repo(
                repo_id=repo_name, repo_type="dataset", token=token, exist_ok=True
            )

            logger.info(f"Uploading data to repository: {repo_name}")

            # Upload files
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_name,
                repo_type="dataset",
                token=token,
                delete_patterns=["*"],  # Delete all existing files before uploading
            )

            logger.info(f"Upload successful! Repository URL: {repo_url}")
            return repo_url

        except Exception as e:
            logger.error(f"Error occurred during upload: {e!s}")
            raise

    def to_modelscope(
        self,
        repo_name: str | None = None,
    ):
        """
        Upload a dataset to the ModelScope Hub

        Args:
            repo_name: Repository name on ModelScope Hub (format: 'username/dataset-name')
        """
        # Initialize API
        local_path = f"{self.storage.root_path}/{self.name}"
        # if storage is not local, download to local
        if self.storage.storage_type != StorageType.LOCAL:
            logger.info("Downloading dataset to local storage")
            self.to_storage(StorageConfig(type=StorageType.LOCAL, root_path=local_path))
        uploader = ModelScopeDatasetUploader(repo_path=repo_name)
        success, error_msg = uploader.upload(str(local_path), commit_message="upload dataset")
        if success:
            repo_url = f"https://www.modelscope.cn/datasets/{repo_name}"
            logger.info(f"Upload successful! Repository URL: {repo_url}")
        else:
            logger.error(f"upload dataset failed: {error_msg}")

    def generate_data(
        self,
        num_rows: int | dict[str, int] | None = None,
        splits: list[str] | None | None = None,
        target_file_size_mb: int = 512,
        num_buffers: int = 15,
        queue_size: int = 30,
        batch_size: int = 100_000,
    ) -> None:
        """Generate synthetic data based on the schema.

        Args:
            num_rows (Union[int, Dict[str, int]], optional): Number of rows to generate.
                Can be either:
                - An integer: Same number of rows for all splits
                - A dictionary: Mapping split names to their row counts, e.g. {"train": 1000, "test": 200}
                Defaults to {"train": 1000, "test": 200}.
            splits (List[str], optional): List of splits to generate data for. If None, uses ["train", "test"].
            target_file_size_mb (int, optional): Target size of each parquet file in MB. Defaults to 512.
            num_buffers (int, optional): Number of buffers for writing. Defaults to 15.
            queue_size (int, optional): Size of the writer queue. Defaults to 30.
        """

        import pandas as pd

        if splits is None:
            splits = ["train", "test"]
        if num_rows is None:
            num_rows = {"train": 3000, "test": 200}
        # Convert num_rows to dictionary if it's an integer
        if isinstance(num_rows, int):
            num_rows = {split: num_rows for split in splits}
        else:
            # Ensure all requested splits have a row count
            for split in splits:
                if split not in num_rows:
                    raise ValueError(
                        f"No row count specified for split '{split}' in num_rows dictionary"
                    )

        for split in splits:
            dataset = self.datasets[split]
            schema = dataset.get_schema()
            if schema is None:
                raise ValueError(f"No schema found for split '{split}'. Please set schema first.")

            split_num_rows = num_rows[split]
            logger.info(f"Generating {split_num_rows} rows for split '{split}'")

            # Use context manager for writer
            with dataset.get_writer(
                mode="overwrite",
                target_file_size_mb=target_file_size_mb,
                num_buffers=num_buffers,
                queue_size=queue_size,
            ) as writer:
                batch_size = min(
                    split_num_rows, batch_size
                )  # Process in batches to avoid memory issues
                for batch_start in range(0, split_num_rows, batch_size):
                    batch_end = min(batch_start + batch_size, split_num_rows)
                    batch_size_actual = batch_end - batch_start
                    data = gen_row_data_by_schema(
                        schema=schema, nb=batch_size_actual, start=batch_start
                    )
                    # Convert to DataFrame and write
                    df = pd.DataFrame(data)
                    t0_write = time.time()
                    logger.info(f"write df: \n{df}")
                    writer.write(df, verify_schema=True)
                    logger.info(
                        f"Write batch {batch_start//batch_size + 1} cost {time.time()-t0_write:.2f}s"
                    )

            logger.info(f"Generated {num_rows[split]} rows of data for split '{split}'")


def list_datasets() -> list[dict[str, str | dict]]:
    config = get_config()
    root_path = config.storage.root_path
    fs = _create_filesystem(config.storage)
    datasets = []

    try:
        for item in fs.ls(root_path):
            if fs.isdir(item):
                dataset_name = Path(item).name
                datasets.append({"name": dataset_name})
    except Exception as e:
        logger.error(f"Error listing datasets: {e!s}")

    return datasets


def load_dataset(
    name: str,
    split: str | list[str] | None = None,
    schema: CollectionSchema | None = None,
) -> Dataset | DatasetDict:
    if split is None:
        splits = ["train", "test", "neighbors"]
        datasets = {
            s: Dataset(
                name,
                split=s,
            )
            for s in splits
        }
        dataset_dict = DatasetDict(datasets)
        if schema:
            dataset_dict["train"].set_schema(schema)
        return dataset_dict
    elif isinstance(split, str):
        dataset = Dataset(name, split=split)
        if schema:
            dataset.set_schema(schema)
        return dataset
    elif isinstance(split, list):
        datasets = {s: Dataset(name, split=s) for s in split}
        dataset_dict = DatasetDict(datasets)
        if schema:
            dataset_dict["train"].set_schema(schema)
        return dataset_dict
    else:
        raise ValueError("split must be None, a string, or a list of strings")
