import glob
import time

import pandas as pd
from loguru import logger
from pymilvus import CollectionSchema, DataType, FieldSchema

from milvus_dataset import ConfigManager, StorageType, load_dataset

logger.info("start to create dataset")


ConfigManager().init_storage(
    root_path="./data/cohere-dataset",
    storage_type=StorageType.LOCAL,
)


id_field = FieldSchema("idx", DataType.INT64, is_primary=True)
chunk_field = FieldSchema("chunk_id", DataType.VARCHAR, max_length=100)
emb_field = FieldSchema("emb", DataType.FLOAT_VECTOR, dim=1024)
url_field = FieldSchema("url", DataType.VARCHAR, max_length=25536)
title_field = FieldSchema("title", DataType.VARCHAR, max_length=25536)
text_field = FieldSchema("text", DataType.VARCHAR, max_length=25536)
schema = CollectionSchema(
    fields=[id_field, chunk_field, url_field, title_field, text_field, emb_field],
    description="我的数据集schema",
)

logger.info(f"schema: {schema}")
logger.info("start to load dataset")
dataset = load_dataset("cohere-v3-generate-demo", schema=schema)
logger.info("succeed to load dataset")
print(dataset)

dataset.generate_data()


dataset.compute_neighbors(
    vector_field_name="emb",
    pk_field_name="idx",
    top_k=1000,
    max_rows_per_epoch=1000000,
    metric_type="cosine",
)
res = dataset.summary()
logger.info(f"summary: {res}")
