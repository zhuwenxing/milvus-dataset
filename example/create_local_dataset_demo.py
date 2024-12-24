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
dataset = load_dataset("cohere-v3-small", schema=schema)
logger.info("succeed to load dataset")
print(dataset)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

file_list = glob.glob("./sample_data/*.parquet")
logger.info(f"file length {len(file_list)}")

df = pd.read_parquet(file_list[0])
logger.info(f"row num each file {len(df)}\n {df}")

train_size = 1000_000
test_size = 100
train_batch_size = 1000
epoch = 1
test_batch_size = test_size // epoch
logger.info(
    f"train_size: {train_size}, test_size: {test_size}, train_batch_size: {train_batch_size}, test_batch_size: {test_batch_size}"
)

test_data_list = []


with train_dataset.get_writer(
    mode="overwrite", target_file_size_mb=512, num_buffers=15, queue_size=30
) as writer:
    for e in range(epoch):
        t0 = time.time()
        t0_read = time.time()
        df = pd.read_parquet(file_list[e])
        logger.info(f"read parquet cost {time.time()-t0_read}")
        df = df.rename(columns={"_id": "chunk_id"})
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        # 创建索引列表
        train_idx_list = list(range(e * train_batch_size, (e + 1) * train_batch_size))
        test_idx_list = list(range(e * test_batch_size, (e + 1) * test_batch_size))  #

        train_data = df_shuffled.iloc[:train_batch_size].copy()
        test_data = df_shuffled.iloc[train_batch_size : train_batch_size + test_batch_size].copy()

        logger.info(
            f"train data shape: {train_data.shape}, train_idx_list length: {len(train_idx_list)}"
        )

        # 使用 .loc 来设置 'idx' 列
        train_data.loc[:, "idx"] = train_idx_list
        test_data.loc[:, "idx"] = test_idx_list

        test_data_list.append(test_data)
        t0_write = time.time()
        writer.write(train_data, mode="append", verify_schema=False)
        logger.info(f"write cost {time.time()-t0_write}")
        tt = time.time() - t0
        logger.info(f"process {e+1}/{epoch} cost {tt}")

with test_dataset.get_writer(
    mode="overwrite", target_file_size_mb=512, num_buffers=15, queue_size=30
) as writer:
    for test_data in test_data_list:
        writer.write(test_data, mode="append", verify_schema=False)

dataset.compute_neighbors(
    vector_field_name="emb",
    pk_field_name="idx",
    top_k=1000,
    max_rows_per_epoch=1000000,
    metric_type="cosine",
)
res = dataset.summary()
logger.info(f"summary: {res}")
