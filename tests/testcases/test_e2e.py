import random
import time
from pymilvus import FieldSchema, CollectionSchema, DataType
from milvus_dataset import list_datasets, load_dataset, ConfigManager, StorageType

config_manager = ConfigManager()
config_manager.init_storage("/tmp/milvus_dataset")


id_field = FieldSchema("id", DataType.INT64, is_primary=True)
vector_field = FieldSchema("vector", DataType.FLOAT_VECTOR, dim=128)
text_field = FieldSchema("text", DataType.VARCHAR, max_length=200)
schema = CollectionSchema(fields=[id_field, vector_field, text_field], description="我的数据集schema")


class TestMilvusDatasetE2E:
    def test_list_datasets(self):
        test_dataset = load_dataset("test_dataset", schema=schema)
        d_list = list_datasets()
        assert "test_dataset" in [d['name']for d in d_list]

    def test_load_dataset(self):
        dataset = load_dataset("test_dataset")
        assert dataset is not None

    def test_read_full(self):
        demo = load_dataset("test_dataset")
        demo['train'].write(
            {
                "id": list(range(1000)),
                "text": [f"text_{i}" for i in range(1000)],
                "vector": [[random.random() for _ in range(128)] for i in range(1000)],
            },
            mode="overwrite",
        )
        demo['test'].write(
            {
                "id": list(range(1000)),
                "text": [f"text_{i}" for i in range(1000)],
                "vector": [[random.random() for _ in range(128)] for i in range(1000)],
            },
            mode="overwrite",
        )
        for name, split in demo.items():
            if name in ['train', 'test']:
                data = split.read(mode="full")
                assert len(data) == 1000

    def test_read_stream(self):
        demo = load_dataset("test_dataset")
        demo['train'].write(
            {
                "id": list(range(1000)),
                "text": [f"text_{i}" for i in range(1000)],
                "vector": [[random.random() for _ in range(128)] for i in range(1000)],
            },
            mode="overwrite",
        )
        demo['test'].write(
            {
                "id": list(range(1000)),
                "text": [f"text_{i}" for i in range(1000)],
                "vector": [[random.random() for _ in range(128)] for i in range(1000)],
            },
            mode="overwrite",
        )
        demo.compute_neighbors(vector_field_name="vector", top_k=10, metric_type="cosine")
        for name, split in demo.items():
            data = split.read(mode="stream")
            for d in data:
                print(d)
                assert len(d) == 1

    def test_read_batch(self):
        def test_read_stream(self):
            demo = load_dataset("test_dataset")
            demo['train'].write(
                {
                    "id": list(range(1000)),
                    "text": [f"text_{i}" for i in range(1000)],
                    "vector": [[random.random() for _ in range(128)] for i in range(1000)],
                },
                mode="overwrite",
            )
            demo['test'].write(
                {
                    "id": list(range(1000)),
                    "text": [f"text_{i}" for i in range(1000)],
                    "vector": [[random.random() for _ in range(128)] for i in range(1000)],
                },
                mode="overwrite",
            )
            demo.compute_neighbors(vector_field_name="vector", top_k=10, metric_type="cosine")
            for name, split in demo.items():
                data = split.read(mode="batch", batch_size=100)
                for d in data:
                    assert len(d) == 100


