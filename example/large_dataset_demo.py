from loguru import logger
from pymilvus import CollectionSchema, DataType, FieldSchema

from milvus_dataset import ConfigManager, StorageType, load_dataset

logger.info("Start to create dataset with all data types")

# Initialize storage
ConfigManager().init_storage(
    root_path="./data/milvus_dataset_demo",
    storage_type=StorageType.LOCAL,
)

# Define schema with all supported data types
schema = CollectionSchema(
    fields=[
        # Primary key
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("int64_val", DataType.INT64),
        # Vector fields
        FieldSchema("float_vector_1", DataType.FLOAT_VECTOR, dim=32),
        FieldSchema("float_vector_2", DataType.FLOAT_VECTOR, dim=32),
    ],
    description="Large Dataset Demo ",
)

logger.info(f"Schema created: {schema}")

# Load dataset
dataset = load_dataset("large_dataset_demo", schema=schema)
logger.info("Dataset loaded successfully")

# Generate data
dataset.generate_data(num_rows={"train": 10_000_000, "test": 1000})

# Compute neighbors for vector fields
dataset.compute_neighbors(
    vector_field_name="float_vector_1",
    pk_field_name="id",
    top_k=10,
    max_rows_per_epoch=1000_000,
    metric_type="l2",
)

dataset.compute_neighbors(
    vector_field_name="float_vector_2",
    pk_field_name="id",
    top_k=10,
    max_rows_per_epoch=1000_000,
    metric_type="l2",
)

# Get summary of the dataset
summary = dataset.summary()
logger.info(f"Dataset summary: {summary}")
# milvus_storage = StorageConfig(
#     root_path="milvus-bucket/import-files",
#     storage_type=StorageType.S3,
#     options={
#         "key": "minioadmin",  # MinIO访问密钥
#         "secret": "minioadmin",  # MinIO密钥
#         "endpoint_url": "http://10.104.34.95:9000",  # MinIO服务器地址
#         "use_ssl": False,  # 如果使用HTTPS则设为True
#     },
# )
# dataset.to_milvus(
#     milvus_config={"uri": "http://10.104.26.252:19530"}, milvus_storage=milvus_storage
# )
