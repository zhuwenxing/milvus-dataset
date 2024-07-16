# tests/test_writer.py

import pytest
import pandas as pd
import pyarrow as pa
from unittest.mock import Mock, patch
from milvus_dataset.writer import DatasetWriter


@pytest.fixture
def mock_dataset():
    dataset = Mock()
    dataset.storage = Mock()
    dataset.root_path = "/tmp"
    dataset.name = "mock_dataset"
    dataset.storage.join.return_value = "/tmp/mock_dataset.parquet"
    dataset.storage.exists.return_value = False
    return dataset


def test_writer_initialization(mock_dataset):
    writer = DatasetWriter(mock_dataset)
    assert writer.dataset == mock_dataset


def test_writer_write_dataframe(mock_dataset):
    writer = DatasetWriter(mock_dataset)
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    writer.write(df)

def test_writer_write_dict(mock_dataset):
    writer = DatasetWriter(mock_dataset)
    data = {
        "id": [1, 2, 3],
        "value": [10, 20, 30]
    }
    writer.write(data)

def test_writer_write_list(mock_dataset):
    writer = DatasetWriter(mock_dataset)
    data = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30}
    ]
    writer.write(data)


def test_writer_write_dataframe_existing_file(mock_dataset):
    writer = DatasetWriter(mock_dataset)
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

    # Mock that the file already exists
    mock_dataset.storage.exists.return_value = True

    # Use unittest.mock.patch
    with patch('pyarrow.parquet.read_table') as mock_read_table, \
            patch('pyarrow.parquet.write_table') as mock_write_table:
        # Mock existing data
        mock_read_table.return_value = pa.Table.from_pandas(pd.DataFrame({"id": [4, 5], "value": [40, 50]}))

        writer._write_dataframe(df, mode="append")

        # Assert that read_table and write_table were called
        mock_read_table.assert_called_once()
        mock_write_table.assert_called_once()

        # Check that the written data is the concatenation of existing and new data
        args, kwargs = mock_write_table.call_args
        written_table = args[0]
        assert written_table.num_rows == 5  # 2 existing + 3 new rows
        assert written_table.column_names == ["id", "value"]

# Add more tests as needed
