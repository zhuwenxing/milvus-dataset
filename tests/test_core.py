import pytest
import os
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from milvus_dataset.core import Dataset, list_datasets
from milvus_dataset.storage import LocalStorage

@pytest.fixture(scope="module")
def test_dir():
    # Create a temporary directory for testing
    test_dir = Path("/tmp/test_datasets")
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Clean up after all tests are done
    shutil.rmtree(test_dir)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'id': range(100),
        'text': [f'text_{i}' for i in range(100)],
        'value': [i * 0.1 for i in range(100)]
    })


def test_dataset_initialization(test_dir):
    dataset = Dataset(str(test_dir), "test_dataset")
    assert dataset.root_path == str(test_dir)
    assert dataset.name == "test_dataset"
    assert isinstance(dataset.storage, LocalStorage)


def test_write_and_read(test_dir, sample_data):
    dataset = Dataset(str(test_dir), "write_read_test")

    # Test write
    dataset.write(sample_data)

    # Test read
    read_data = dataset.read(mode='full')
    pd.testing.assert_frame_equal(read_data, sample_data)

    # Test stream read
    stream_data = pd.concat(list(dataset.read(mode='stream')), ignore_index=True)
    assert stream_data.to_dict() == sample_data.to_dict()
    # Test batch read
    batch_data = pd.concat(list(dataset.read(mode='batch', batch_size=10)), ignore_index=True)
    assert batch_data.to_dict() == sample_data.to_dict()


def test_metadata(test_dir):
    dataset = Dataset(str(test_dir), "metadata_test")

    # Test metadata saving
    dataset.metadata['test_key'] = 'test_value'
    dataset._save_metadata()

    # Test metadata loading
    new_dataset = Dataset(str(test_dir), "metadata_test")
    assert new_dataset.metadata['test_key'] == 'test_value'


def test_summary(test_dir, sample_data):
    dataset = Dataset(str(test_dir), "summary_test")
    dataset.write(sample_data)

    summary = dataset.summary()
    assert summary['name'] == "summary_test"
    assert summary['num_rows'] == 100
    assert summary['num_columns'] == 3
    assert set(summary['schema'].keys()) == {'id', 'text', 'value'}


def test_list_datasets(test_dir):
    # Create multiple datasets
    Dataset(str(test_dir), "dataset1").write(pd.DataFrame({'a': [1, 2, 3]}))
    Dataset(str(test_dir), "dataset2").write(pd.DataFrame({'b': [4, 5, 6]}))

    datasets = list_datasets(str(test_dir))
    assert len(datasets) == 2
    assert {'dataset1', 'dataset2'} == {d['name'] for d in datasets}


@pytest.mark.parametrize("embedding_type", ['dense', 'sparse'])
def test_generate_embeddings(test_dir, sample_data, embedding_type):
    dataset = Dataset(str(test_dir), f"{embedding_type}_embedding_test")
    dataset.write(sample_data)

    if embedding_type == 'dense':
        embeddings = dataset.generate_dense_embeddings('text', 'sentence-transformers/all-MiniLM-L6-v2')
    else:
        embeddings = dataset.generate_sparse_embeddings('text', 'tfidf')

    assert len(embeddings) == len(sample_data)


def test_compute_neighbors(test_dir, sample_data):
    dataset = Dataset(str(test_dir), "neighbors_test")
    dataset.write(sample_data)

    # Generate embeddings first
    embeddings = dataset.generate_dense_embeddings('text', 'sentence-transformers/all-MiniLM-L6-v2')

    # Compute neighbors
    query = embeddings[0].reshape(1, -1)  # Use the first embedding as a query
    distances, indices = dataset.compute_neighbors(query)

    assert len(distances) > 0
    assert len(indices) > 0


def test_generate_scalar_distribution(test_dir):
    dataset = Dataset(str(test_dir), "distribution_test")
    distribution = dataset.generate_scalar_distribution('normal(0,1)', hit_rate=0.5, size=1000)

    assert len(distribution) == 1000
    assert 400 < distribution.count() < 600  # Roughly 50% hit rate

# Add more tests as needed


