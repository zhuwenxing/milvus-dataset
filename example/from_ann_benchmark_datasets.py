import logging
import os
import urllib.request

import h5py
from pymilvus import CollectionSchema, DataType, FieldSchema

from milvus_dataset import ConfigManager, StorageType, load_dataset

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ConfigManager().init_storage(
    root_path="./data/anns-benchmark-dataset",
    storage_type=StorageType.LOCAL,
)

# Define datasets to download
DATASETS = ["kosarak-jaccard", "sift-256-hamming"]

# Create download directory
DOWNLOAD_DIR = "anns_datasets/ann_datasets"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_dataset(dataset_name):
    """Download a dataset from ann-benchmarks.com"""
    url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
    output_file = os.path.join(DOWNLOAD_DIR, f"{dataset_name}.hdf5")

    print(f"Downloading {dataset_name}...")
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"✅ Successfully downloaded: {dataset_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {dataset_name}: {e!s}")
        if os.path.exists(output_file):
            os.remove(output_file)
        return False


# Download all datasets
# for dataset in DATASETS:
#     download_dataset(dataset)


def transform_ann_dataset_to_milvus_dataset(file_path):
    dataset_name = file_path.split("/")[-1].split(".")[0]
    if not any(x in dataset_name for x in ["hamming"]):
        return
    dim = None
    with h5py.File(file_path, "r") as f:
        for name, dataset in f.items():
            if isinstance(dataset, h5py.Dataset) and name in ["train", "test"]:
                print(f"\nDataset '{name}':")
                print(f"Shape: {dataset.shape}")
                print(f"Data type: {dataset.dtype}")
                dim = dataset.shape[1]
    id_field = FieldSchema("idx", DataType.INT64, is_primary=True)
    emb_field = FieldSchema("emb", DataType.BINARY_VECTOR, dim=dim)
    schema = CollectionSchema(
        fields=[id_field, emb_field],
        description=dataset_name,
    )
    dict = schema.to_dict()
    logger.info(f"schema dict: {dict}")
    dataset = load_dataset(dataset_name, schema=schema)
    dataset.set_metadata({"source": "https://github.com/erikbern/ann-benchmarks"})
    if any(x in dataset_name for x in ["angular"]):
        # needs to use cosine similarity and normalize the vectors
        metric_type = "cosine"
    elif any(x in dataset_name for x in ["dot"]):
        metric_type = "inner_product"
    elif any(x in dataset_name for x in ["euclidean"]):
        metric_type = "l2"
    elif any(x in dataset_name for x in ["hamming"]):
        metric_type = "hamming"
    elif any(x in dataset_name for x in ["jaccard"]):
        metric_type = "jaccard"

    with h5py.File(file_path, "r") as f:
        train_data = f["train"]
        test_data = f["test"]

        with dataset["train"].get_writer(mode="overwrite") as writer:
            writer.write({"idx": list(range(train_data.shape[0])), "emb": list(train_data)})
        with dataset["test"].get_writer(mode="overwrite") as writer:
            writer.write({"idx": list(range(test_data.shape[0])), "emb": list(test_data)})

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
    file_path = os.path.join(DOWNLOAD_DIR, f"{dataset_name}.hdf5")
    if os.path.exists(file_path):
        print(f"\nProcessing {dataset_name}...")
        try:
            transform_ann_dataset_to_milvus_dataset(file_path)
            print(f"Successfully processed {dataset_name}")
        except Exception as e:
            print(f"Error processing {dataset_name}: {e!s}")
    else:
        print(f"Dataset file not found: {file_path}")
