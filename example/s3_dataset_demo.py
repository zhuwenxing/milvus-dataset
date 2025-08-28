import dotenv
from loguru import logger
from pymilvus import CollectionSchema, DataType, FieldSchema

from milvus_dataset import ConfigManager, StorageType, load_dataset

logger.info("start to create dataset")
config_manager = ConfigManager()


dotenv.load_dotenv()

key = dotenv.get_key(".env", "AWS_ACCESS_KEY_ID")
secret = dotenv.get_key(".env", "AWS_SECRET_ACCESS_KEY")
root_path = dotenv.get_key(".env", "ROOT_PATH")

# MinIO配置
options = {
    "key": key,  # MinIO访问密钥
    "secret": secret,  # MinIO密钥
    "region_name": "us-west-2",
}

ConfigManager().init_storage(root_path=root_path, storage_type=StorageType.S3, options=options)


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
dataset.generate_data(num_rows={"train": 3_000, "test": 1000})

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
