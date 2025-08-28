from pymilvus import CollectionSchema, DataType, FieldSchema

from milvus_dataset import ConfigManager, StorageType, load_dataset

config_manager = ConfigManager()
# config_manager.init_storage("/tmp/milvus_dataset")


if __name__ == "__main__":
    import sys

    print(sys.version)
    ConfigManager().init_storage(
        root_path="./data",
        storage_type=StorageType.LOCAL,
    )

    # 创建schema
    id_field = FieldSchema("id", DataType.INT64, is_primary=True)
    vector_field = FieldSchema("emb", DataType.FLOAT_VECTOR, dim=128)
    schema = CollectionSchema(fields=[id_field, vector_field], description="我的数据集schema")

    dataset = load_dataset("mongodb-test", schema=schema)
    print(dataset)
    dataset.generate_data(num_rows={"train": 5_000, "test": 1000})
    dataset.compute_neighbors(
        pk_field_name="id",
        vector_field_name="emb",
        top_k=1000,
        max_rows_per_epoch=10000,
        metric_type="cosine",
    )
    dataset.summary()
    print(dataset)
