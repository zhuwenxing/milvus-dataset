import random
from pymilvus import FieldSchema, CollectionSchema, DataType
import time
from pymilvus import MilvusClient
from milvus_dataset import list_datasets, load_dataset, ConfigManager, StorageType, StorageConfig
from loguru import logger

config_manager = ConfigManager()
# config_manager.init_storage("/tmp/milvus_dataset")


if __name__ == "__main__":
    import sys

    print(sys.version)
    ConfigManager().init_storage(
        root_path="/tmp/milvus_dataset",
        storage_type=StorageType.LOCAL,
    )

    # 创建schema
    id_field = FieldSchema("id", DataType.INT64, is_primary=True)
    vector_field = FieldSchema("emb", DataType.FLOAT_VECTOR, dim=128)
    text_field = FieldSchema("text", DataType.VARCHAR, max_length=200)
    schema = CollectionSchema(fields=[id_field, vector_field, text_field], description="我的数据集schema")

    dataset = load_dataset("openai_large_demo_2", schema=schema)
    print(dataset)
    train_data = dataset['train']
    dataset_size = 50000
    data = {
        "id": range(dataset_size),
        "text": [f"text_{i}" for i in range(dataset_size)],
        "emb": [[random.random() for _ in range(128)] for i in range(dataset_size)],
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
            "emb": [[random.random() for _ in range(128)] for i in range(dataset_size)],
        },
        mode="overwrite"
    )
    dataset.compute_neighbors(vector_field_name="emb", top_k=100, max_rows_per_epoch=10000, metric_type="cosine")

    print(dataset)

    milvus_config = {
        "uri": 'http://10.104.33.77:19530',
        "token": 'root:Milvus',
        "db_name": 'default'
    }

    milvus_storage = StorageConfig(
        root_path="s3://milvus-bucket/milvus-dataset",
        type=StorageType.S3,
        options={
            "key": "minioadmin",
            "secret": "minioadmin",
            "endpoint_url": "http://10.104.33.79:9000",
            "use_ssl": False,
        }
    )
    logger.info("Start to save dataset to milvus")
    dataset.to_milvus(milvus_config, "import", milvus_storage=milvus_storage)
