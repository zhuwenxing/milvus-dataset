import glob
import time

import pandas as pd
from loguru import logger
from pymilvus import CollectionSchema, DataType, FieldSchema

from milvus_dataset import ConfigManager, StorageType, load_dataset, list_datasets

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
dict =schema.to_dict()
logger.info(f"schema dict: {dict}")


all_datasets = list_datasets()
logger.info(f"all datasets: {all_datasets}")
