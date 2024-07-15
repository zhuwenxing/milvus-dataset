# tests/test_core.py
import pytest
from milvus_dataset import Dataset

def test_dataset_initialization():
    dataset = Dataset(root_path="/tmp", name="test_dataset")
    assert dataset.name == "test_dataset"
    assert dataset.root_path == "/tmp"

def test_dataset_write_and_read():
    dataset = Dataset(root_path="/tmp", name="test_dataset")
    test_data = {"id": [1, 2, 3], "value": [10, 20, 30]}
    dataset.write(test_data)
    read_data = dataset.read(mode="full")
    assert len(read_data) == 3
    assert list(read_data["id"]) == [1, 2, 3]

@pytest.mark.parametrize("invalid_input", [None, 42, "string"])
def test_dataset_write_invalid_input(invalid_input):
    dataset = Dataset(root_path="/tmp", name="test_dataset")
    with pytest.raises(ValueError):
        dataset.write(invalid_input)
