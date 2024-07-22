import json
import threading
from pathlib import Path

import fsspec
from fsspec.spec import AbstractFileSystem
from enum import Enum
from pydantic import BaseModel, create_model
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threading import Lock

from .logging import logger
from .writer import DatasetWriter
from .reader import DatasetReader
from .neighbors import NeighborsComputation

class StorageType(Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gs"


class StorageConfig(BaseModel):
    type: StorageType
    root_path: str
    options: Dict[str, Any] = {}


class DatasetConfig(BaseModel):
    storage: StorageConfig
    default_schema: Optional[Dict[str, Any]] = None


class ConfigManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance.config = None
        return cls._instance

    def get_config(self) -> DatasetConfig:
        if self.config is None:
            raise ValueError("ConfigManager has not been initialized")
        return self.config

    def init_storage(self, root_path: str, storage_type: StorageType = StorageType.LOCAL, **options):
        if storage_type == StorageType.S3:
            options = self._prepare_s3_options(options)
            self._verify_s3_connection(root_path, options)
        config = DatasetConfig(
            storage=StorageConfig(
                type=storage_type,
                root_path=root_path,
                options=options
            )
        )
        self._initialize(config)

    def _prepare_s3_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        s3_options = {
            "key": options.get("aws_access_key_id"),
            "secret": options.get("aws_secret_access_key"),
            "client_kwargs": {}
        }

        if "endpoint_url" in options:
            s3_options["client_kwargs"]["endpoint_url"] = options["endpoint_url"]
        if "region_name" in options:
            s3_options["client_kwargs"]["region_name"] = options["region_name"]
        if "use_ssl" in options:
            s3_options["use_ssl"] = options["use_ssl"]

        return s3_options

    def _verify_s3_connection(self, root_path: str, options: Dict[str, Any]):
        try:
            fs = fsspec.filesystem("s3", **options)
            bucket = root_path.split("://")[1].split("/")[0]
            fs.ls(bucket)
            logger.info(f"Successfully connected to bucket: {bucket}")
        except Exception as e:
            logger.error(f"Failed to connect to S3/MinIO: {str(e)}")
            raise

    def set_default_schema(self, schema: Dict[str, Any]):
        if self.config is None:
            raise ValueError("Storage configuration must be initialized before setting schema")
        new_config = DatasetConfig(
            storage=self.config.storage,
            default_schema=schema
        )
        self._initialize(new_config)

    def _initialize(self, config: DatasetConfig):
        with self._lock:
            self.config = config


def get_config() -> DatasetConfig:
    return ConfigManager().get_config()


def _create_filesystem(storage_config: StorageConfig) -> AbstractFileSystem:
    logger.info(f"Creating filesystem with config: {storage_config.dict()}")
    if storage_config.type == StorageType.LOCAL:
        return fsspec.filesystem("file")
    elif storage_config.type == StorageType.S3:
        try:
            fs = fsspec.filesystem("s3", **storage_config.options)
            # 测试连接
            bucket = storage_config.root_path.split("://")[1].split("/")[0]
            fs.ls(bucket)
            return fs
        except Exception as e:
            logger.error(f"Failed to create S3 filesystem: {str(e)}")
            raise
    elif storage_config.type == StorageType.GCS:
        return fsspec.filesystem("gcs", **storage_config.options)
    else:
        raise ValueError(f"Unsupported storage type: {storage_config.type}")


class Dataset:
    def __init__(self, name: str, schema: Dict = None, split="train", metadata=None):
        self.name = name
        self.config = get_config()
        self.fs = _create_filesystem(self.config.storage)
        self.root_path = self.config.storage.root_path
        self._schema = None
        self.metadata = self._load_metadata()
        self.split = split
        logger.info(f"Initializing dataset '{name}' with split '{split}'")
        if self.split == "train":
            self._set_or_load_schema(schema)
        self._ensure_split_exists()
        self.writer = DatasetWriter(self)
        self.reader = DatasetReader(self)

    def _set_or_load_schema(self, schema: Optional[Dict[str, Any]]):
        schema_path = f"{self.root_path}/{self.name}/schema.json"
        if schema:
            with self.fs.open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2)
            self._schema = create_model('DataSchema', **schema)
        elif self.fs.exists(schema_path):
            with self.fs.open(schema_path, 'r') as f:
                schema_dict = json.load(f)
            self._schema = create_model('DataSchema', **schema_dict)
        else:
            self._schema = None

    def _ensure_split_exists(self):
        split_dir = f"{self.root_path}/{self.name}/{self.split}"
        self.fs.makedirs(split_dir, exist_ok=True)

    def _load_metadata(self):
        metadata_path = f"{self.root_path}/{self.name}/metadata.json"
        if self.fs.exists(metadata_path):
            with self.fs.open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        metadata_path = f"{self.root_path}/{self.name}/metadata.json"
        with self.fs.open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def write(self, data: Union[pd.DataFrame, Dict, List[Dict]], mode: str = 'append', verify_schema: bool = False):
        logger.info(f"Writing data to dataset '{self.name}'")
        if verify_schema:
            self._schema.validate(data)
        return self.writer.write(data, mode)

    def read(self, mode: str = 'stream', batch_size: int = 1000):
        return self.reader.read(mode, batch_size)

    def get_total_rows(self, split: str) -> int:
        # 实现此方法以返回指定 split 的总行数
        pass

    def summary(self) -> Dict[str, Union[str, int, Dict]]:
        path = f"{self.root_path}/{self.name}"
        if not self.fs.exists(path):
            return {
                "name": self.name,
                "size": 0,
                "num_rows": 0,
                "num_columns": 0,
                "schema": {},
                "storage_type": self.config.storage.type.value,
                "num_files": 0
            }

        total_rows = 0
        total_size = 0
        schema_dict = {}
        num_files = 0

        for file in self.fs.glob(f"{path}/*.parquet"):
            num_files += 1
            with self.fs.open(file, 'rb') as f:
                parquet_file = pq.ParquetFile(f)
                total_rows += parquet_file.metadata.num_rows
                if not schema_dict:
                    schema = parquet_file.schema.to_arrow_schema()
                    schema_dict = {field.name: str(field.type) for field in schema}
                total_size += self.fs.info(file)['size']

        return {
            "name": self.name,
            "size": total_size,
            "num_rows": total_rows,
            "num_columns": len(schema_dict),
            "schema": schema_dict,
            "storage_type": self.config.storage.type.value,
            "num_files": num_files
        }


class DatasetDict(dict):
    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__(datasets)

    def __getitem__(self, key: str) -> Dataset:
        return super().__getitem__(key)

    def compute_neighbors(self, vector_field_name, query_expr=None, top_k=1000, **kwargs):
        neighbors_computation = NeighborsComputation(self, vector_field_name, query_expr=query_expr, top_k=top_k, **kwargs)
        neighbors_computation.compute_ground_truth()


def list_datasets() -> List[Dict[str, Union[str, Dict]]]:
    config = get_config()
    root_path = config.storage.root_path
    fs = _create_filesystem(config.storage)
    datasets = []

    try:
        for item in fs.ls(root_path):
            if fs.isdir(item):
                dataset_name = Path(item).name
                metadata_path = f"{item}/{dataset_name}_metadata.json"
                metadata = {}
                if fs.exists(metadata_path):
                    with fs.open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                datasets.append({"name": dataset_name, "metadata": metadata})
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")

    return datasets


def load_dataset(name: str, split: Optional[Union[str, List[str]]] = None, mode='batch', batch_size=10000) -> Union[Dataset, DatasetDict]:
    """
    Load a dataset from the given root path.

    Args:
        name (str): Name of the dataset
        split (Optional[Union[str, List[str]]]): Split(s) to load
        mode (str): Read mode ('batch' or 'stream')
        batch_size (int): Batch size for reading

    Returns:
        Union[Dataset, DatasetDict]: The loaded dataset(s)
    """
    available_splits = ['train', 'test', 'neighbors']

    if split is None:
        return DatasetDict({
            s: Dataset(name, split=s) for s in available_splits
        })
    elif isinstance(split, str):
        return Dataset(name, split=split)
    elif isinstance(split, list):
        return DatasetDict({
            s: Dataset(name, split=s) for s in split
        })
    else:
        raise ValueError("Split must be None, a string, or a list of strings")
