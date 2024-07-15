# tests/test_reader.py
import pytest
from milvus_dataset.reader import DatasetReader

@pytest.fixture
def mock_dataset():
    class MockDataset:
        def __init__(self):
            self.storage = None
    return MockDataset()

def test_reader_initialization(mock_dataset):
    reader = DatasetReader(mock_dataset)
    assert reader.dataset == mock_dataset

def test_reader_read_full(mock_dataset):
    reader = DatasetReader(mock_dataset)
    # Mock the necessary methods and test full read
    # Add assertions based on expected behavior
