import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema

from milvus_dataset import ConfigManager, StorageType, list_datasets, load_dataset


class TestLocalDatasetE2E:
    @classmethod
    def setup_class(cls):
        # Create a temporary directory for testing
        cls.temp_dir = tempfile.mkdtemp(prefix="milvus_dataset_test_")

        # Initialize storage with local path
        config_manager = ConfigManager()
        config_manager.init_storage(
            root_path=cls.temp_dir,
            storage_type=StorageType.LOCAL,
        )

        # Create schema matching the example
        cls.id_field = FieldSchema("idx", DataType.INT64, is_primary=True)
        cls.chunk_field = FieldSchema("chunk_id", DataType.VARCHAR, max_length=100)
        cls.emb_field = FieldSchema("emb", DataType.FLOAT_VECTOR, dim=1024)
        cls.url_field = FieldSchema("url", DataType.VARCHAR, max_length=25536)
        cls.title_field = FieldSchema("title", DataType.VARCHAR, max_length=25536)
        cls.text_field = FieldSchema("text", DataType.VARCHAR, max_length=25536)

        cls.schema = CollectionSchema(
            fields=[
                cls.id_field,
                cls.chunk_field,
                cls.url_field,
                cls.title_field,
                cls.text_field,
                cls.emb_field,
            ],
            description="Test dataset schema",
        )

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory after tests"""
        import shutil

        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_storage_initialization(self):
        """Test if storage is properly initialized"""
        config = ConfigManager().get_config()
        assert config.storage.root_path == self.temp_dir
        assert config.storage.storage_type == StorageType.LOCAL
        assert os.path.exists(self.temp_dir)

    def test_dataset_creation_and_listing(self):
        """Test creating datasets and listing them"""
        # Create multiple datasets
        dataset_names = [f"test_dataset_{i}" for i in range(3)]
        for name in dataset_names:
            load_dataset(name, schema=self.schema)

        # List all datasets
        all_datasets = list_datasets()
        listed_names = [d["name"] for d in all_datasets]

        # Verify all created datasets are listed
        for name in dataset_names:
            assert name in listed_names

    def test_write_and_read_data(self):
        """Test writing and reading data from the dataset"""
        dataset_name = "test_write_read_dataset"
        dataset = load_dataset(dataset_name, schema=self.schema)

        # Prepare test data
        num_samples = 10
        test_data = {
            "idx": list(range(num_samples)),
            "chunk_id": [f"chunk_{i}" for i in range(num_samples)],
            "url": [f"http://example.com/{i}" for i in range(num_samples)],
            "title": [f"Title {i}" for i in range(num_samples)],
            "text": [f"Sample text {i}" for i in range(num_samples)],
            "emb": [np.random.rand(1024).tolist() for _ in range(num_samples)],
        }

        # Write data using context manager
        with dataset["train"].get_writer(mode="overwrite") as writer:
            df = pd.DataFrame(test_data)
            writer.write(df, verify_schema=True)

        # Read and verify data
        loaded_data = dataset["train"].read(mode="full")
        assert len(loaded_data) == num_samples
        assert all(isinstance(id_, int | np.integer) for id_ in loaded_data["idx"])
        assert all(isinstance(chunk_id, str) for chunk_id in loaded_data["chunk_id"])
        assert all(isinstance(url, str) for url in loaded_data["url"])
        assert all(isinstance(title, str) for title in loaded_data["title"])
        assert all(isinstance(text, str) for text in loaded_data["text"])
        assert all(len(emb) == 1024 for emb in loaded_data["emb"])

    def test_multiple_splits(self):
        """Test handling multiple splits in the dataset"""
        dataset_name = "test_splits_dataset"
        dataset = load_dataset(dataset_name, schema=self.schema)

        splits = ["train", "test"]
        samples_per_split = {"train": 100, "test": 20}

        # Write different amounts of data to each split
        for split in splits:
            num_samples = samples_per_split[split]
            test_data = {
                "idx": list(range(num_samples)),
                "chunk_id": [f"{split}_chunk_{i}" for i in range(num_samples)],
                "url": [f"http://example.com/{split}/{i}" for i in range(num_samples)],
                "title": [f"{split.capitalize()} Title {i}" for i in range(num_samples)],
                "text": [f"{split.capitalize()} text {i}" for i in range(num_samples)],
                "emb": [np.random.rand(1024).tolist() for _ in range(num_samples)],
            }
            with dataset[split].get_writer(mode="overwrite") as writer:
                df = pd.DataFrame(test_data)
                writer.write(df, verify_schema=True)

        # Verify each split
        for split in splits:
            loaded_data = dataset[split].read(mode="full")
            assert len(loaded_data) == samples_per_split[split]
            assert all(
                chunk_id.startswith(f"{split}_chunk_") for chunk_id in loaded_data["chunk_id"]
            )

    def test_batch_reading(self):
        """Test reading data in batches"""
        dataset_name = "test_batch_dataset"
        dataset = load_dataset(dataset_name, schema=self.schema)

        # Write 150 samples
        num_samples = 150
        test_data = {
            "idx": list(range(num_samples)),
            "chunk_id": [f"chunk_{i}" for i in range(num_samples)],
            "url": [f"http://example.com/{i}" for i in range(num_samples)],
            "title": [f"Title {i}" for i in range(num_samples)],
            "text": [f"Sample text {i}" for i in range(num_samples)],
            "emb": [np.random.rand(1024).tolist() for _ in range(num_samples)],
        }

        # Write data using context manager
        with dataset["train"].get_writer(mode="overwrite") as writer:
            df = pd.DataFrame(test_data)
            writer.write(df, verify_schema=True)

        # Read in batches of 50
        batch_size = 50
        total_samples = 0
        for batch in dataset["train"].read(mode="batch", batch_size=batch_size):
            assert len(batch) <= batch_size
            total_samples += len(batch)

        assert total_samples == num_samples

    @pytest.mark.parametrize("expr", [None, "idx < 5"])
    def test_ground_truth_computation(self, expr):
        """Test computing ground truth neighbors"""
        dataset_name = "test_ground_truth_dataset"
        dataset = load_dataset(dataset_name, schema=self.schema)

        # Create train and test data with controlled embeddings for predictable neighbors
        train_samples = 100
        test_samples = 20
        dim = 1024

        # Create train data with known patterns
        train_data = {
            "idx": list(range(train_samples)),
            "chunk_id": [f"train_chunk_{i}" for i in range(train_samples)],
            "url": [f"http://example.com/train/{i}" for i in range(train_samples)],
            "title": [f"Train Title {i}" for i in range(train_samples)],
            "text": [f"Train text {i}" for i in range(train_samples)],
            "emb": [],
        }

        # Create test data
        test_data = {
            "idx": list(range(test_samples)),
            "chunk_id": [f"test_chunk_{i}" for i in range(test_samples)],
            "url": [f"http://example.com/test/{i}" for i in range(test_samples)],
            "title": [f"Test Title {i}" for i in range(test_samples)],
            "text": [f"Test text {i}" for i in range(test_samples)],
            "emb": [],
        }

        # Create embeddings with known similarities
        np.random.seed(42)  # For reproducibility
        base_vectors = np.random.rand(5, dim)  # Create 5 base vectors

        # Create train embeddings as variations of base vectors
        for i in range(train_samples):
            base_idx = i % 5
            noise = np.random.rand(dim) * 0.1  # Small random noise
            vec = base_vectors[base_idx] + noise
            vec = vec / np.linalg.norm(vec)  # Normalize
            train_data["emb"].append(vec.tolist())

        # Create test embeddings as variations of the same base vectors
        for i in range(test_samples):
            base_idx = i % 5
            noise = np.random.rand(dim) * 0.1  # Small random noise
            vec = base_vectors[base_idx] + noise
            vec = vec / np.linalg.norm(vec)  # Normalize
            test_data["emb"].append(vec.tolist())

        # Write data using context manager
        with dataset["train"].get_writer(mode="overwrite") as writer:
            df = pd.DataFrame(train_data)
            writer.write(df, verify_schema=True)

        with dataset["test"].get_writer(mode="overwrite") as writer:
            df = pd.DataFrame(test_data)
            writer.write(df, verify_schema=True)

        # Compute neighbors
        top_k = 10
        dataset.compute_neighbors(
            vector_field_name="emb",
            pk_field_name="idx",
            top_k=top_k,
            metric_type="cosine",
            query_expr=expr,
        )

        # Get and verify neighbors
        neighbors_data = dataset.get_neighbors("emb", pk_field_name="idx", query_expr=expr)
        assert not neighbors_data.empty, "Neighbors data should not be empty"

        # Verify basic properties of neighbors
        assert len(neighbors_data) == test_samples  # One row per test sample
