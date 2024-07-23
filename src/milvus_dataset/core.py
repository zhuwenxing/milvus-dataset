import json
import threading
from pathlib import Path
import numpy as np
import fsspec
import botocore
from fsspec.spec import AbstractFileSystem
from enum import Enum
from pymilvus import FieldSchema, CollectionSchema, DataType
from pydantic import BaseModel, create_model
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import Schema
from threading import Lock
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
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
            "key": options.get("aws_access_key_id") or options.get("access_key") or options.get("key"),
            "secret": options.get("aws_secret_access_key") or options.get("secret_key") or options.get("secret"),
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

            try:
                fs.ls(bucket)
                logger.info(f"Successfully connected to existing bucket: {bucket}")
            except Exception as e:
                try:
                    fs.mkdir(bucket)
                    logger.info(f"Successfully created and connected to new bucket: {bucket}")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket {bucket}: {str(create_error)}")
                    raise

        except Exception as e:
            logger.error(f"Failed to connect to S3/MinIO: {str(e)}")
            raise

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
            try:
                fs.ls(bucket)
                logger.info(f"Successfully connected to existing bucket: {bucket}")
            except Exception as e:
                try:
                    fs.mkdir(bucket)
                    logger.info(f"Successfully created and connected to new bucket: {bucket}")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket {bucket}: {str(create_error)}")
                    raise
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

    def set_schema(self, schema: CollectionSchema):
        """设置数据集的schema。"""
        self._schema = schema
        self._save_schema()
        logger.info(f"已为数据集 '{self.name}' 设置schema")

    def _save_schema(self):
        """将schema保存到文件。"""
        schema_path = f"{self.root_path}/{self.name}/schema.json"
        with self.fs.open(schema_path, 'w') as f:
            json.dump(self._schema.to_dict(), f, indent=2)

    def _load_schema(self):
        """如果存在,从文件加载schema。"""
        schema_path = f"{self.root_path}/{self.name}/schema.json"
        if self.fs.exists(schema_path):
            with self.fs.open(schema_path, 'r') as f:
                schema_dict = json.load(f)
            return CollectionSchema.construct_from_dict(schema_dict)
        return None

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

    def get_schema(self) -> Optional[CollectionSchema]:
        """获取数据集当前的schema。"""
        if self._schema is None:
            self._schema = self._load_schema()
        return self._schema

    def _verify_schema(self, data: Union[pd.DataFrame, Dict, List[Dict]]):
        # if column is auto id, remove it from schema
        # column 不能多，也不能少

        if isinstance(data, pd.DataFrame):
            for field in self._schema.fields:
                if field.auto_id:
                    continue
                if field.name not in data.columns:
                    raise ValueError(f"数据中缺少字段 '{field.name}'。")
        elif isinstance(data, dict) or (isinstance(data, list) and isinstance(data[0], dict)):
            sample = data if isinstance(data, dict) else data[0]
            for field in self._schema.fields:
                if field.name not in sample:
                    raise ValueError(f"数据中缺少字段 '{field.name}'。")

    def _get_summary(self) -> Dict[str, Union[str, int, Dict]]:
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
                    "storage_type": self.config.storage.type.value,
                    "num_files": 0
                }
            else:
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

                self._summary = {
                    "name": self.name,
                    "split": self.split,
                    "size": total_size,
                    "num_rows": total_rows,
                    "num_columns": len(schema_dict),
                    "schema": schema_dict,
                    "storage_type": self.config.storage.type.value,
                    "num_files": num_files
                }

        return self._summary

    def get_features(self) -> Dict[str, str]:
        """
        获取数据集的特征（schema）。

        返回:
            Dict[str, str]: 一个字典，键是字段名，值是数据类型的字符串表示。
        """
        summary = self._get_summary()
        return summary['schema']

    def get_num_rows(self) -> int:
        """
        获取数据集的行数。

        返回:
            int: 数据集的总行数。
        """
        summary = self._get_summary()
        return summary['num_rows']

    def get_num_columns(self) -> int:
        """
        获取数据集的列数。

        返回:
            int: 数据集的列数。
        """
        summary = self._get_summary()
        return summary['num_columns']

    def get_size(self) -> int:
        """
        获取数据集的大小（字节）。

        返回:
            int: 数据集的总大小（字节）。
        """
        summary = self._get_summary()
        return summary['size']

    def get_num_files(self) -> int:
        """
        获取数据集的文件数。

        返回:
            int: 数据集的文件数。
        """
        summary = self._get_summary()
        return summary['num_files']

    def write(self, data: Union[pd.DataFrame, Dict, List[Dict]], mode: str = 'append', verify_schema: bool = False):
        logger.info(f"正在向数据集 '{self.name}' 写入数据")
        if self._schema is None:
            self._schema = self._load_schema()
        if self._schema is None:
            raise ValueError("写入数据前必须设置schema。请使用set_schema()方法。")
        if self.split not in ['train', 'test']:
            raise ValueError("只允许向'train'和'test'分割写入数据。")
        if verify_schema:
            self._verify_schema(data)
        result = self.writer.write(data, mode)
        self._summary = None
        return result

    def read(self, mode: str = 'stream', batch_size: int = 1000):
        return self.reader.read(mode, batch_size)

    def get_total_rows(self, split: str) -> int:
        # 实现此方法以返回指定 split 的总行数
        pass

    def summary(self) -> Dict[str, Union[str, int, Dict]]:
        path = f"{self.root_path}/{self.name}/{self.split}"
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
            "size": f"{total_size / 1024 / 1024:.3f} MB",
            "num_rows": total_rows,
            "num_columns": len(schema_dict),
            "schema": schema_dict,
            "storage_type": self.config.storage.type.value,
            "num_files": num_files
        }


class DatasetDict(dict):
    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__(datasets)
        self.datasets = datasets
        self.name = datasets['train'].name
        self.train = datasets['train']

    def __getitem__(self, key: str) -> Dataset:
        return super().__getitem__(key)

    def save(self, destination: StorageConfig):
        """
        Save the dataset by copying all files to a specified destination.

        Args:
            destination (StorageConfig): The storage configuration for the destination where the dataset should be saved.
        """
        dest_fs = _create_filesystem(destination)

        for split, dataset in self.datasets.items():
            source_path = f"{dataset.root_path}/{dataset.name}/{split}"
            dest_path = f"{destination.root_path}/{dataset.name}/{split}"

            # Ensure the destination directory exists
            dest_fs.makedirs(dest_path, exist_ok=True)

            # Copy all files from source to destination
            for file in dataset.fs.glob(f"{source_path}/*.parquet"):
                file_name = Path(file).name
                with dataset.fs.open(file, 'rb') as source_file:
                    with dest_fs.open(f"{dest_path}/{file_name}", 'wb') as dest_file:
                        dest_file.write(source_file.read())

            logger.info(f"Saved {split} split to {dest_path}")

        # Copy metadata and schema files
        metadata_file = f"{self.datasets['train'].root_path}/{self.name}/metadata.json"
        schema_file = f"{self.datasets['train'].root_path}/{self.name}/schema.json"

        for file in [metadata_file, schema_file]:
            if self.datasets['train'].fs.exists(file):
                file_name = Path(file).name
                dest_file_path = f"{destination.root_path}/{self.name}/{file_name}"
                with self.datasets['train'].fs.open(file, 'rb') as source_file:
                    with dest_fs.open(dest_file_path, 'wb') as dest_file:
                        dest_file.write(source_file.read())
                logger.info(f"Saved {file_name} to {dest_file_path}")

        logger.info(f"Dataset '{self.name}' has been successfully saved to {destination.root_path}")

    def summary(self) -> Dict:
        """
        获取整个数据集字典的摘要信息。

        返回:
            Dict: 包含所有分割摘要信息的字典
        """
        return {split: dataset.summary() for split, dataset in self.datasets.items()}

    def __repr__(self):
        dataset_dict = self.summary()

        # 创建最终的字典结构
        final_dict = {"DatasetDict": dataset_dict}

        # 使用 json.dumps 进行格式化，缩进设置为 2 个空格
        return json.dumps(final_dict, indent=2)

    def to_dict(self):
        """返回数据集字典的原始字典表示"""
        return self.summary()

    def compute_neighbors(self, vector_field_name, query_expr=None, top_k=1000, **kwargs):
        neighbors_computation = NeighborsComputation(self, vector_field_name, query_expr=query_expr, top_k=top_k,
                                                     **kwargs)
        neighbors_computation.compute_ground_truth()

    def get_neighbors(self, query_expr=None):
        neighbors = self['neighbors']
        file_name = f"{neighbors.root_path}/{neighbors.name}/{neighbors.split}/neighbors-{query_expr}.parquet"
        if neighbors.fs.exists(file_name):
            with neighbors.fs.open(file_name, 'rb') as f:
                return pq.read_table(f).to_pandas()
        else:
            logger.warning(f"Neighbors file not found: {file_name}")
            return pd.DataFrame()

    def set_schema(self, schema: CollectionSchema):
        """为所有分割设置schema。"""
        for dataset in self.values():
            dataset.set_schema(schema)
        logger.info(f"已为数据集 '{self.name}' 的所有分割设置schema")

    def to_milvus(self, host: str = "localhost", port: str = "19530", collection_name: str = None,
                  id_field: str = None, vector_field: str = None, index=None):
        """
        Transfer the dataset to a Milvus collection, automatically generating the schema from the DataFrame.

        Args:
            collection_name (str): Name of the Milvus collection to create or use
            host (str): Milvus server host
            port (str): Milvus server port
            id_field (str): Name of the field to use as the primary key (optional)
            vector_field (str): Name of the field containing the vector data (optional)

        Returns:
            None
        """
        if collection_name is None:
            collection_name = self.name
        logger.info(f"Transferring dataset '{self.name}' to Milvus collection '{collection_name}'")

        # Connect to Milvus
        connections.connect(host=host, port=port)

        # Read the first batch of data to infer the schema
        first_batch = next(self.train.read(mode='stream'))

        # Generate Milvus schema from DataFrame
        fields = self._generate_milvus_schema(first_batch, id_field, vector_field)

        # Check if collection exists, if not, create it
        if not utility.has_collection(collection_name):
            schema = CollectionSchema(fields, description=f"Collection for dataset {self.name}")
            collection = Collection(name=collection_name, schema=schema)
            logger.info(f"Created new Milvus collection: {collection_name}")
        else:
            collection = Collection(name=collection_name)
            logger.info(f"Using existing Milvus collection: {collection_name}")

        # Insert data
        for batch in self.train.read(mode='stream'):
            self._insert_data(collection, batch, fields)

        # Flush the collection to ensure all data is written
        collection.flush()
        if index is None:
            index = {
                "index_type": "FLAT",
                "metric_type": "L2",
                "params": {},
            }
        collection.create_index(vector_field, index)

        # Load the collection for search
        collection.load()

        logger.info(f"Successfully transferred dataset '{self.name}' to Milvus collection '{collection_name}'")

    def _generate_milvus_schema(self, df: pd.DataFrame, id_field: Optional[str] = None,
                                vector_field: Optional[str] = None) -> List[FieldSchema]:
        fields = []
        for column, dtype in df.dtypes.items():
            column_name = str(column)  # Convert column name to string
            if column_name == id_field or (id_field is None and column_name == str(df.index.name)):
                fields.append(FieldSchema(name=column_name, dtype=DataType.INT64, is_primary=True, auto_id=False))
            elif column_name == vector_field or (
                    vector_field is None and dtype == 'object' and isinstance(df[column].iloc[0], (list, np.ndarray))):
                dim = len(df[column].iloc[0])
                fields.append(FieldSchema(name=column_name, dtype=DataType.FLOAT_VECTOR, dim=dim))
            elif np.issubdtype(dtype, np.integer):
                fields.append(FieldSchema(name=column_name, dtype=DataType.INT64))
            elif np.issubdtype(dtype, np.floating):
                fields.append(FieldSchema(name=column_name, dtype=DataType.FLOAT))
            elif dtype == 'bool':
                fields.append(FieldSchema(name=column_name, dtype=DataType.BOOL))
            else:
                fields.append(FieldSchema(name=column_name, dtype=DataType.VARCHAR, max_length=65535))
        return fields

    def _insert_data(self, collection: Collection, df: pd.DataFrame, fields: List[FieldSchema]):
        entities = []
        for _, row in df.iterrows():
            entity = {}
            for field in fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    entity[field.name] = row[field.name].tolist()
                else:
                    entity[field.name] = row[field.name]
            entities.append(entity)
        collection.insert(entities)


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


def load_dataset(name: str, split: Optional[Union[str, List[str]]] = None, schema: Optional[CollectionSchema] = None) -> \
Union[Dataset, DatasetDict]:
    if split is None:
        splits = ['train', 'test', 'neighbors']
        datasets = {s: Dataset(name, split=s, ) for s in splits}
        dataset_dict = DatasetDict(datasets)
        if schema:
            dataset_dict.train.set_schema(schema)
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
            dataset_dict.train.set_schema(schema)
        return dataset_dict
    else:
        raise ValueError("split 必须是 None、字符串或字符串列表")
