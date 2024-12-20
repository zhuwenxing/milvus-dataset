from loguru import logger

from milvus_dataset import ConfigManager, StorageType, load_dataset

logger.info("start to create dataset")
config_manager = ConfigManager()
# config_manager.init_storage("./data/cohere-v3-1M")


ConfigManager().init_storage(
    root_path="./data/cohere-v3-1M",
    storage_type=StorageType.LOCAL,
)

dataset = load_dataset("cohere-v3-10M")
logger.info("succeed to load dataset")
print(dataset)
dataset.to_hf(repo_name=f"WenxingZhu/{dataset.name}")
