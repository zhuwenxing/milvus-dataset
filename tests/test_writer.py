# tests/test_writer.py
import pytest
import pandas as pd
from src.milvus_dataset.writer import DatasetWriter

@pytest.fixture
def mock_dataset():
    class MockDataset:
        def __init__(self):
            self.storage = None
    return MockDataset()

def test_writer_initialization(mock_dataset):
    writer = DatasetWriter(mock_dataset)
    assert writer.dataset == mock_dataset

def test_writer_write_dataframe(mock_dataset):
    writer = DatasetWriter(mock_dataset)
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    result = writer._write_dataframe(df, mode="append")
    # Add assertions based on expected behavior
