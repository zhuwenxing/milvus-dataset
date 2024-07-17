import json
import os
from pathlib import Path
import pandas as pd
from enum import Enum
from pydantic import BaseModel, create_model, ValidationError
from typing import Union, Dict, List, Any, Optional
import pyarrow.parquet as pq
from threading import Lock
from filelock import FileLock
import threading
import time
from .logging import logger
from .storage import StorageBackend, LocalStorage, S3Storage
from .models.schema import DatasetSchema
from .writer import DatasetWriter
from .reader import DatasetReader
from .milvus.operations import MilvusOperations
from .embeddings.dense import generate_dense_embeddings
from .embeddings.sparse import generate_sparse_embeddings
from .operations.neighbors import compute_neighbors
from .utils.distribution import generate_scalar_distribution


class StorageType(Enum):
    LOCAL = "local"
    S3 = "s3"

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

    def init_local_storage(self, root_path: str):
        config = DatasetConfig(
            storage=StorageConfig(
                type=StorageType.LOCAL,
                root_path=root_path
            )
        )
        self._initialize(config)

    def init_s3_storage(self, bucket: str, access_key: str, secret_key: str, region: str = "us-west-2"):
        config = DatasetConfig(
            storage=StorageConfig(
                type=StorageType.S3,
                root_path=f"s3://{bucket}",
                options={
                    "aws_access_key_id": access_key,
                    "aws_secret_access_key": secret_key,
                    "region_name": region
                }
            )
        )
        self._initialize(config)

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


# Global function to get the config
def get_config() -> DatasetConfig:
    return ConfigManager().get_config()


def _create_storage(storage_config: StorageConfig):
    if storage_config.type == StorageType.LOCAL:
        return LocalStorage(storage_config.root_path)
    elif storage_config.type == StorageType.S3:
        return S3Storage(storage_config.root_path, **storage_config.options)
    else:
        raise ValueError(f"Unsupported storage type: {storage_config.type}")


class Dataset:
    def __init__(self, name: str, schema: Dict = None, split="train", metadata=None):
        self.name = name
        self.config = get_config()
        self.storage = _create_storage(self.config.storage)
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
        schema_path = os.path.join(self.config.storage.root_path, self.name, "schema.json")
        if schema:
            with open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2)
            self._schema = create_model('DataSchema', **schema)
        elif os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema_dict = json.load(f)
            self._schema = create_model('DataSchema', **schema_dict)
        else:
            self._schema = None

    def _ensure_split_exists(self):
        dataset_dir = os.path.join(self.root_path, self.name)
        split_dir = os.path.join(dataset_dir, self.split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir, exist_ok=True)

    def _load_metadata(self):
        metadata_path = self.storage.join(self.root_path, f"{self.name}/metadata.json")
        if self.storage.exists(metadata_path):
            return self.storage.read_json(metadata_path)
        return {}

    def _save_metadata(self):
        metadata_path = self.storage.join(self.root_path, f"{self.name}/metadata.json")
        self.storage.write_json(metadata_path, self.metadata)

    def write(self, data: Union[pd.DataFrame, Dict, List[Dict]], mode: str = 'append', verify_schema: bool = False):
        # validate data
        logger.info(f"Writing data to dataset '{self.name}'")
        if verify_schema:
            self._schema.validate(data)
        return self.writer.write(data, mode)

    def read(self, mode: str = 'stream', batch_size: int = 1000):
        return self.reader.read(mode, batch_size)

    def generate_dense_embeddings(self, text_column: str, model_name: str):
        return generate_dense_embeddings(self, text_column, model_name)

    def generate_sparse_embeddings(self, text_column: str, model_name: str):
        return generate_sparse_embeddings(self, text_column, model_name)

    def compute_neighbors(self, query_data, filter_condition=None, use_gpu=False):
        return compute_neighbors(self, query_data, filter_condition, use_gpu)

    def generate_scalar_distribution(self, expression: str, hit_rate: float, size: int):
        return generate_scalar_distribution(expression, hit_rate, size)

    def summary(self) -> Dict[str, Union[str, int, Dict]]:
        """
        Generate a summary of the dataset.

        Returns:
            Dict containing summary information about the dataset.
        """
        dataset_path = Path(self.storage.join(self.root_path, self.name))
        if not self.storage.exists(str(dataset_path)):
            return {
                "name": self.name,
                "size": 0,
                "num_rows": 0,
                "num_columns": 0,
                "schema": {},
                "storage_type": type(self.storage).__name__,
                "num_files": 0
            }

        total_rows = 0
        total_size = 0
        schema_dict = {}
        num_files = 0

        for file in dataset_path.glob('*.parquet'):
            num_files += 1
            parquet_file = pq.ParquetFile(str(file))

            # Accumulate total rows
            total_rows += parquet_file.metadata.num_rows

            # Get schema information (assuming all files have the same schema)
            if not schema_dict:
                schema = parquet_file.schema.to_arrow_schema()
                schema_dict = {field.name: str(field.type) for field in schema}

            # Get file size
            if isinstance(self.storage, LocalStorage):
                file_size = os.path.getsize(file)
            elif isinstance(self.storage, S3Storage):
                response = self.storage.client.head_object(
                    Bucket=self.storage.bucket,
                    Key=self.storage._get_full_path(str(file.relative_to(self.root_path)))
                )
                file_size = response['ContentLength']
            else:
                file_size = None  # Unable to determine file size for unknown storage types

            if file_size is not None:
                total_size += file_size

        return {
            "name": self.name,
            "size": total_size,
            "num_rows": total_rows,
            "num_columns": len(schema_dict),
            "schema": schema_dict,
            "storage_type": type(self.storage).__name__,
            "num_files": num_files
        }

class DatasetDict(dict):
    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__(datasets)

class FileLock:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.lock = threading.Lock()

    def __enter__(self):
        self.lock.acquire()
        while os.path.exists(self.lock_file):
            time.sleep(0.1)
        open(self.lock_file, 'w').close()

    def __exit__(self, type, value, traceback):
        os.remove(self.lock_file)
        self.lock.release()


def list_datasets() -> List[
    Dict[str, Union[str, Dict]]]:
    config = get_config()
    root_path = config.storage.root_path
    storage = _create_storage(config.storage)
    datasets = []

    if isinstance(storage, LocalStorage) or storage is None:
        root_path = Path(root_path)
        root_path.mkdir(parents=True, exist_ok=True)
        for item in root_path.iterdir():
            if item.is_dir():
                dataset_name = item.name
                metadata_path = item / f"{dataset_name}_metadata.json"
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                datasets.append({"name": dataset_name, "metadata": metadata})

    elif isinstance(storage, S3Storage):
        # List all "folders" in S3
        response = storage.client.list_objects_v2(
            Bucket=storage.bucket,
            Prefix=storage.prefix,
            Delimiter='/'
        )

        for prefix in response.get('CommonPrefixes', []):
            folder_path = prefix.get('Prefix', '')
            dataset_name = folder_path.rstrip('/').split('/')[-1]

            # Try to get metadata file if it exists
            metadata_key = f"{folder_path}{dataset_name}_metadata.json"
            try:
                metadata_content = storage.read(metadata_key)
                metadata = json.loads(metadata_content.decode('utf-8'))
            except Exception:
                metadata = {}  # Metadata file doesn't exist or couldn't be read

            datasets.append({"name": dataset_name, "metadata": metadata})

    else:
        raise NotImplementedError(f"list_datasets not implemented for {type(storage).__name__}")

    return datasets





def load_dataset(name: str, split: Optional[Union[str, List[str]]] = None, mode='batch', batch_size=10000):
    """
    Load a dataset from the given root path.

    Args:
        :param name:
        :param mode:
        :param batch_size:

    Returns:
        Dataset: The loaded dataset.
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

