from loguru import logger

from milvus_dataset import ConfigManager, StorageConfig, StorageType, load_dataset

logger.info("start to create dataset")


# local storage
ConfigManager().init_storage(
    root_path="./data/cohere-dataset",
    storage_type=StorageType.LOCAL,
)


dataset = load_dataset("cohere-v3-small")
logger.info("succeed to load dataset")
print(dataset)


# minio storage
minio_storage = StorageConfig(
    root_path="milvus-dataset/benchmark-dataset-to-storage-v2",
    storage_type=StorageType.S3,
    options={
        "key": "minioadmin",  # MinIO访问密钥
        "secret": "minioadmin",  # MinIO密钥
        "endpoint_url": "http://10.104.34.95:9000",  # MinIO服务器地址
        "use_ssl": False,  # 如果使用HTTPS则设为True
    },
)


dataset.to_storage(minio_storage)
