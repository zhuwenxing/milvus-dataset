
import fsspec
from fsspec.spec import AbstractFileSystem
from enum import Enum
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union, List
from .logging import logger

class StorageType(Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gs"


class StorageConfig(BaseModel):
    type: StorageType
    root_path: str
    options: Dict[str, Any] = {}



def _create_filesystem(storage_config: StorageConfig) -> AbstractFileSystem:
    logger.info(f"Creating filesystem with config: {storage_config.dict()}")
    if storage_config.type == StorageType.LOCAL:
        return fsspec.filesystem("file")
    elif storage_config.type == StorageType.S3:
        try:
            fs = fsspec.filesystem("s3", **storage_config.options)
            # 测试连接
            bucket = storage_config.root_path.split("://")[1].split("/")[0]
            try:
                fs.ls(bucket)
                logger.info(f"Successfully connected to existing bucket: {bucket}")
            except Exception as e:
                try:
                    fs.mkdir(bucket)
                    logger.info(f"Successfully created and connected to new bucket: {bucket}")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket {bucket}: {str(create_error)}")
                    raise
            return fs
        except Exception as e:
            logger.error(f"Failed to create S3 filesystem: {str(e)}")
            raise
    elif storage_config.type == StorageType.GCS:
        return fsspec.filesystem("gcs", **storage_config.options)
    else:
        raise ValueError(f"Unsupported storage type: {storage_config.type}")
