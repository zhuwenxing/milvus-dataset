import random
import time
from pymilvus import FieldSchema, CollectionSchema, DataType
from milvus_dataset import list_datasets, load_dataset, ConfigManager, StorageType

config_manager = ConfigManager()
# config_manager.init_storage("/tmp/milvus_dataset")


ConfigManager().init_storage(
    root_path="/tmp/milvus_dataset",
    storage_type=StorageType.LOCAL,
)


id_field = FieldSchema("id", DataType.INT64, is_primary=True)
vector_field = FieldSchema("emb", DataType.FLOAT_VECTOR, dim=768)
text_field = FieldSchema("text", DataType.VARCHAR, max_length=200)
schema = CollectionSchema(fields=[id_field, vector_field, text_field], description="我的数据集schema")


dataset = load_dataset("openai_large_1m_v2", schema=schema)
print(dataset)
train_data = dataset['train']
dataset_size = 1000000
batch_size = 1000000
epoch = dataset_size// batch_size
for e in range(epoch):
    data = {
            "id": range(batch_size),
            "text": [f"text_{i}" for i in range(batch_size)],
            "emb": [[random.random() for _ in range(768)] for i in range(batch_size)],
        }
    train_data.write(
        data,
        mode="overwrite",
        verify_schema=True
    )
test_data = dataset['test']
dataset_size = 1000
test_data.write(
    {
        "id": range(dataset_size),
        "text": [f"text_{i}" for i in range(dataset_size)],
        "emb": [[random.random() for _ in range(768)] for i in range(dataset_size)],
    },
    mode="overwrite",
)

print("Start computing neighbors")
t0 = time.time()
dataset.compute_neighbors(vector_field_name="emb", top_k=2000, max_rows_per_epoch=1000000, metric_type="cosine")
tt = time.time() - t0
print(f"compute neighbors cost {tt}")
data = dataset.get_neighbors(query_expr=None)
print(data)




