import logging
import os

from datasets import load_dataset as load_hf_dataset
from pymilvus import CollectionSchema, DataType, FieldSchema

from milvus_dataset import load_dataset as load_milvus_dataset

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Define datasets to download
DATASETS = [
    "Cohere/wikipedia-22-12-simple-embeddings",
    # etc.
]

# Create download directory
DOWNLOAD_DIR = "huggingface_datasets"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_dataset(dataset_name):
    """Download a dataset from huggingface.co"""

    print(f"Downloading {dataset_name}...")
    try:
        load_hf_dataset(dataset_name, split="all", cache_dir=DOWNLOAD_DIR)
        print(f"✅ Successfully downloaded: {dataset_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {dataset_name}: {e!s}")
        return False


# Download all datasets
for dataset in DATASETS:
    download_dataset(dataset)


def transform_hf_dataset_to_milvus_dataset(dataset_name, metric_type="cosine"):
    dim = None
    for doc in load_hf_dataset(dataset_name, split="train"):
        if dim is None:
            dim = len(doc["emb"])
        else:
            assert dim == len(doc["emb"])
        break

    id_field = FieldSchema("id", DataType.INT64, is_primary=True)
    emb_field = FieldSchema("emb", DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(
        fields=[id_field, emb_field],
        description=dataset_name,
    )
    dict = schema.to_dict()
    logger.info(f"schema dict: {dict}")
    dataset = load_milvus_dataset(dataset_name, schema=schema)
    dataset.set_metadata({"source": f"https://huggingface.co/datasets/{dataset_name}"})

    with dataset["train"].get_writer(mode="overwrite") as writer:
        writer.write({"idx": list(doc["id"]), "emb": list(doc["emb"])})

    dataset.compute_neighbors(
        vector_field_name="emb",
        pk_field_name="idx",
        top_k=1000,
        max_rows_per_epoch=30000,
        metric_type=metric_type,
    )
    res = dataset.summary()
    logger.info(f"summary: {res}")


# Process all downloaded datasets
for dataset_name in DATASETS:
    transform_hf_dataset_to_milvus_dataset(dataset_name)
