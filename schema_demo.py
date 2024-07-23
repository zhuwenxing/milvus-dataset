from pymilvus import FieldSchema, CollectionSchema, DataType
import random

from milvus_dataset import list_datasets, load_dataset, ConfigManager, StorageType

config_manager = ConfigManager()

ConfigManager().init_storage(
    root_path="/tmp/milvus_dataset",
    storage_type=StorageType.LOCAL,
)
# 创建schema
id_field = FieldSchema("id", DataType.INT64, is_primary=True)
vector_field = FieldSchema("vector", DataType.FLOAT_VECTOR, dim=128)
text_field = FieldSchema("text", DataType.VARCHAR, max_length=200)
schema = CollectionSchema(fields=[id_field, vector_field, text_field], description="我的数据集schema")

# 加载数据集并设置schema
dataset = load_dataset("my_dataset", schema=schema)

print(dataset)


print(dataset)
train_data = dataset['train']
dataset_size = 50000
data = {
        "id": range(dataset_size),
        "text": [f"text_{i}" for i in range(dataset_size)],
        "vector": [[random.random() for _ in range(128)] for i in range(dataset_size)],
    }
train_data.write(
    data,
    mode="overwrite",
)
test_data = dataset['test']
dataset_size = 2000
test_data.write(
    {
        "id": range(dataset_size),
        "text": [f"text_{i}" for i in range(dataset_size)],
        "vector": [[random.random() for _ in range(128)] for i in range(dataset_size)],
    },
    mode="overwrite",
)

print("Start computing neighbors")

dataset.compute_neighbors(vector_field_name="vector", top_k=10, max_rows_per_epoch=10000, metric_type="cosine")

data = dataset.get_neighbors(query_expr=None)
print(data)
print(dataset)

