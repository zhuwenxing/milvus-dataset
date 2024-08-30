import random
import time
from pymilvus import FieldSchema, CollectionSchema, DataType
from milvus_dataset import list_datasets, load_dataset, ConfigManager, StorageType
from loguru import logger
config_manager = ConfigManager()
config_manager.init_storage("/tmp/milvus_dataset")

dim = 768

id_field = FieldSchema("id", DataType.INT64, is_primary=True)
vector_field = FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim)
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

logger.info(config_manager)
d_list = list_datasets()
logger.info(d_list)

dataset = load_dataset("test_dataset", schema=schema)
logger.info(dataset)

train_set = dataset['train']
batch_size = 10000
t0 = time.time()
data = {
    "id": list(range(batch_size)),
    "text": [f"text_{i}" for i in range(batch_size)],
    "vector": [[random.random() for _ in range(dim)] for i in range(batch_size)],
}
tt = time.time() - t0
logger.info(f"Data generation time: {tt:.2f} s")


with train_set.get_writer(mode="overwrite", target_file_size_mb=512, num_buffers=4) as writer:
    writer.write(data)
    for i in range(1, 100):
        writer.write(data, mode="append", verify_schema=False)
demo = load_dataset("test_dataset")
logger.info(demo)
