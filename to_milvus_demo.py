import random
import time

from milvus_dataset import list_datasets, load_dataset, ConfigManager, StorageType

config_manager = ConfigManager()
# config_manager.init_storage("/tmp/milvus_dataset")


ConfigManager().init_storage(
    root_path="/tmp/milvus_dataset",
    storage_type=StorageType.LOCAL,
)

dataset = load_dataset("openai_large_v12")
print(dataset)
train_data = dataset['train']
dataset_size = 50000
data = {
        "id": range(dataset_size),
        "text": [f"text_{i}" for i in range(dataset_size)],
        "emb": [[random.random() for _ in range(32)] for i in range(dataset_size)],
    }
train_data.write(
    data,
    mode="append",
)
test_data = dataset['test']
dataset_size = 2000
test_data.write(
    {
        "id": range(dataset_size),
        "text": [f"text_{i}" for i in range(dataset_size)],
        "emb": [[random.random() for _ in range(32)] for i in range(dataset_size)],
    },
    mode="append",
)
dataset.compute_neighbors(vector_field_name="emb", top_k=100, max_rows_per_epoch=10000, metric_type="cosine")
dataset.to_milvus(
    host="10.104.17.73",
    port="19530",
    id_field="id",
    vector_field="emb",
    collection_name="openai_large_v12",
    index={"index_type": "FLAT", "metric_type": "COSINE"},
)







