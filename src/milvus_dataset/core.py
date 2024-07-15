import json
import os

import pandas as pd
from typing import Union, Dict, List
import pyarrow.parquet as pq
from .storage import StorageBackend, LocalStorage, S3Storage
from .models.schema import DatasetSchema
from .writer import DatasetWriter
from .reader import DatasetReader
from .milvus.operations import MilvusOperations
from .embeddings.dense import generate_dense_embeddings
from .embeddings.sparse import generate_sparse_embeddings
from .operations.neighbors import compute_neighbors
from .utils.distribution import generate_scalar_distribution


class Dataset:
    def __init__(self, root_path: str, name: str, storage_options: Dict = None):
        self.root_path = root_path
        self.name = name
        self.storage = StorageBackend.create(root_path, storage_options)
        self.metadata = self._load_metadata()
        self.schema = DatasetSchema(self.metadata.get('fields_schema', {}))
        self.writer = DatasetWriter(self)
        self.reader = DatasetReader(self)
        self.milvus_ops = MilvusOperations(self)

    def _load_metadata(self):
        metadata_path = self.storage.join(self.root_path, f"{self.name}_metadata.json")
        if self.storage.exists(metadata_path):
            return self.storage.read_json(metadata_path)
        return {}

    def _save_metadata(self):
        metadata_path = self.storage.join(self.root_path, f"{self.name}_metadata.json")
        self.storage.write_json(metadata_path, self.metadata)

    def write(self, data: Union[pd.DataFrame, Dict, List[Dict]], mode: str = 'append'):
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
        dataset_path = self.storage.join(self.root_path, f"{self.name}.parquet")

        if not self.storage.exists(dataset_path):
            return {
                "name": self.name,
                "size": 0,
                "num_rows": 0,
                "num_columns": 0,
                "schema": {},
                "storage_type": type(self.storage).__name__
            }

        # Read metadata from Parquet file
        parquet_file = pq.ParquetFile(dataset_path)

        # Get schema information
        schema = parquet_file.schema.to_arrow_schema()
        schema_dict = {field.name: str(field.type) for field in schema}

        # Get file size
        if isinstance(self.storage, LocalStorage):
            import os
            file_size = os.path.getsize(dataset_path)
        elif isinstance(self.storage, S3Storage):
            response = self.storage.client.head_object(Bucket=self.storage.bucket,
                                                       Key=self.storage._get_full_path(f"{self.name}.parquet"))
            file_size = response['ContentLength']
        else:
            file_size = None  # Unable to determine file size for unknown storage types

        return {
            "name": self.name,
            "size": file_size,
            "num_rows": parquet_file.num_rows,
            "num_columns": len(schema_dict),
            "schema": schema_dict,
            "storage_type": type(self.storage).__name__
        }

    @classmethod
    def list_datasets(cls, root_path: str, storage_options: Dict = None) -> List[Dict[str, Union[str, Dict]]]:
        """
        List all datasets in the given root path.

        Args:
            root_path (str): The root path to search for datasets.
            storage_options (Dict, optional): Options for the storage backend.

        Returns:
            List of dictionaries, each containing information about a dataset.
        """
        storage = StorageBackend.create(root_path, storage_options)
        datasets = []

        if isinstance(storage, LocalStorage):
            for filename in os.listdir(root_path):
                if filename.endswith("_metadata.json"):
                    dataset_name = filename[:-14]  # Remove "_metadata.json"
                    metadata_path = os.path.join(root_path, filename)
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    datasets.append({"name": dataset_name, "metadata": metadata})
        elif isinstance(storage, S3Storage):
            response = storage.client.list_objects_v2(Bucket=storage.bucket, Prefix=storage.prefix)
            for obj in response.get('Contents', []):
                if obj['Key'].endswith("_metadata.json"):
                    dataset_name = obj['Key'].split('/')[-1][:-14]  # Remove "_metadata.json"
                    metadata = json.loads(storage.read(obj['Key']).decode('utf-8'))
                    datasets.append({"name": dataset_name, "metadata": metadata})
        else:
            raise NotImplementedError(f"list_datasets not implemented for {type(storage).__name__}")

        return datasets


def list_datasets(root_path: str, storage_options: Dict = None) -> List[Dict[str, Union[str, Dict]]]:
    """
    List all datasets in the given root path.

    This is a standalone function that provides the same functionality as the class method.

    Args:
        root_path (str): The root path to search for datasets.
        storage_options (Dict, optional): Options for the storage backend.

    Returns:
        List of dictionaries, each containing information about a dataset.
    """
    return Dataset.list_datasets(root_path, storage_options)
