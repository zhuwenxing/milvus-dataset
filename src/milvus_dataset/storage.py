
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


import os
from typing import List, Tuple


def copy_between_filesystems(
        source_fs: AbstractFileSystem,
        dest_fs: AbstractFileSystem,
        source_path: str,
        dest_path: str,
        ignore_patterns: List[str] = None
) -> List[Tuple[str, str, str]]:
    """
    Copy files and directories from one filesystem to another.

    Args:
    source_fs (AbstractFileSystem): Source filesystem
    dest_fs (AbstractFileSystem): Destination filesystem
    source_path (str): Path in the source filesystem to copy from
    dest_path (str): Path in the destination filesystem to copy to
    ignore_patterns (List[str], optional): List of patterns to ignore

    Returns:
    List[Tuple[str, str, str]]: List of (source_path, dest_path, status) for each copied file
    """
    results = []

    def should_ignore(path: str) -> bool:
        if ignore_patterns:
            return any(pattern in path for pattern in ignore_patterns)
        return False

    def copy_file(src: str, dst: str) -> str:
        try:
            with source_fs.open(src, 'rb') as source_file:
                with dest_fs.open(dst, 'wb') as dest_file:
                    dest_file.write(source_file.read())
            return "Success"
        except Exception as e:
            logger.error(f"Failed to copy {src} to {dst}: {str(e)}")
            return f"Failed: {str(e)}"

    def copy_recursive(src_path: str, dst_path: str):
        if should_ignore(src_path):
            return

        if source_fs.isdir(src_path):
            if not dest_fs.exists(dst_path):
                dest_fs.mkdir(dst_path)
            for item in source_fs.ls(src_path):
                s = os.path.join(src_path, os.path.basename(item))
                d = os.path.join(dst_path, os.path.basename(item))
                copy_recursive(s, d)
        else:
            status = copy_file(src_path, dst_path)
            results.append((src_path, dst_path, status))

    copy_recursive(source_path, dest_path)
    return results


# 使用示例
def copy_data(source_config: StorageConfig, dest_config: StorageConfig,
              source_path: str, dest_path: str,
              ignore_patterns: List[str] = None) -> List[Tuple[str, str, str]]:
    source_fs = _create_filesystem(source_config)
    dest_fs = _create_filesystem(dest_config)

    return copy_between_filesystems(source_fs, dest_fs, source_path, dest_path, ignore_patterns)

