from loguru import logger
from pymilvus import CollectionSchema, DataType, FieldSchema

from milvus_dataset import ConfigManager, StorageConfig, StorageType, load_dataset

logger.info("Start to create dataset with all data types")

# Initialize storage
ConfigManager().init_storage(
    root_path="./data/all-types-dataset",
    storage_type=StorageType.LOCAL,
)

# Define schema with all supported data types
schema = CollectionSchema(
    fields=[
        # Primary key
        FieldSchema("id", DataType.INT64, is_primary=True),
        # Basic numeric types
        # FieldSchema("int8_val", DataType.INT8),
        # FieldSchema("int16_val", DataType.INT16),
        # FieldSchema("int32_val", DataType.INT32),
        FieldSchema("int64_val", DataType.INT64),
        # FieldSchema("float_val", DataType.FLOAT),
        # FieldSchema("double_val", DataType.DOUBLE),
        #
        # # String and Boolean
        # FieldSchema("varchar_val", DataType.VARCHAR, max_length=500),
        # FieldSchema("bool_val", DataType.BOOL),
        # JSON field
        FieldSchema("json_val", DataType.JSON),
        # Array field
        FieldSchema("array_val", DataType.ARRAY, element_type=DataType.INT64, max_capacity=10),
        # Vector fields
        FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=8),
        FieldSchema("bf16_vector", DataType.BFLOAT16_VECTOR, dim=8),
        FieldSchema("sparse_vector", DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema("binary_vector", DataType.BINARY_VECTOR, dim=256),
    ],
    description="Dataset with all PyMilvus data types",
)

logger.info(f"Schema created: {schema}")

# Load dataset
dataset = load_dataset("all_types_demo", schema=schema)
logger.info("Dataset loaded successfully")

# Generate data
dataset.generate_data()

# Compute neighbors for vector fields
dataset.compute_neighbors(
    vector_field_name="float_vector",
    pk_field_name="id",
    top_k=10,
    max_rows_per_epoch=1000,
    metric_type="l2",
)

# Get summary of the dataset
summary = dataset.summary()
logger.info(f"Dataset summary: {summary}")
milvus_storage = StorageConfig(
    root_path="milvus-bucket/import-files",
    storage_type=StorageType.S3,
    options={
        "key": "minioadmin",  # MinIO访问密钥
        "secret": "minioadmin",  # MinIO密钥
        "endpoint_url": "http://10.104.34.95:9000",  # MinIO服务器地址
        "use_ssl": False,  # 如果使用HTTPS则设为True
    },
)
dataset.to_milvus(
    milvus_config={"uri": "http://10.104.26.252:19530"}, milvus_storage=milvus_storage
)
