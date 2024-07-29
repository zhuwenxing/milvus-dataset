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


#
# ConfigManager().init_storage(
#     root_path="s3://public-dataset",
#     storage_type=StorageType.S3,
#     aws_access_key_id="minioadmin",
#     aws_secret_access_key="minioadmin",
#     endpoint_url="http://127.0.0.1:9000",
#     use_ssl=False
# )

print(config_manager)
d_list = list_datasets()
print(d_list)

dataset = load_dataset("test_dataset", schema=schema)
print(dataset)

train_data = dataset['train']
dataset_size = 40100
t0 = time.time()
data = {
    "id": range(dataset_size),
    "text": [f"text_{i}" for i in range(dataset_size)],
    "vector": [[random.random() for _ in range(128)] for i in range(dataset_size)],
}

tt = time.time() - t0
print(f"Data generation time: {tt:.2f} s")
train_data.write(
    data,
    mode="overwrite",
)

demo = load_dataset("test_dataset")
print(demo)
for name, split in demo.items():
    print(name)
    print(split.read())
    data = split.read(mode="full")
    print(data)
    for d in data:
        print(d)
        break

for name, split in demo.items():
    print(name)
    print(split.read())
    data = split.read(mode="stream")
    print(data)
    for d in data:
        print(d)
        break
for name, split in demo.items():
    print(name)
    print(split.read())
    data = split.read(mode="batch", batch_size=10000)
    print(data)
    for d in data:
        print(d)
        break
