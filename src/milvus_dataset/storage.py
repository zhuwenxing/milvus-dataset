import os
from enum import Enum
from typing import Any

import fsspec
from fsspec.spec import AbstractFileSystem
from pydantic import BaseModel

from .log_config import logger


class StorageType(Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"


class StorageConfig(BaseModel):
    storage_type: StorageType
    root_path: str
    options: dict[str, Any] | None = None


def _create_filesystem(storage_config: StorageConfig) -> AbstractFileSystem:
    if storage_config.storage_type == StorageType.LOCAL:
        return fsspec.filesystem("file")
    elif storage_config.storage_type == StorageType.S3:
        try:
            fs = fsspec.filesystem("s3", **storage_config.options)
            # Test connection
            # Parse bucket name from both s3://bucket/path and bucket/path formats
            root_path = storage_config.root_path.rstrip("/")
            if root_path.startswith("s3://"):
                # Format: s3://bucket/path -> extract bucket
                path_parts = root_path.split("/")
                bucket = path_parts[2] if len(path_parts) > 2 else path_parts[-1]
            else:
                # Format: bucket/path -> extract bucket
                bucket = root_path.split("/")[0]
            print(bucket)
            try:
                files = fs.ls(bucket)
                logger.info(f"bucket files: {files}")
                logger.info(f"Successfully connected to existing bucket: {bucket}")
            except Exception:
                try:
                    fs.mkdir(bucket)
                    logger.info(f"Successfully created and connected to new bucket: {bucket}")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket {bucket}: {create_error!s}")
                    raise
            return fs
        except Exception as e:
            logger.error(f"Failed to create S3 filesystem: {e!s}")
            raise
    elif storage_config.storage_type == StorageType.GCS:
        return fsspec.filesystem("gcs", **storage_config.options)
    else:
        raise ValueError(f"Unsupported storage type: {storage_config.storage_type}")


def copy_between_filesystems(
    source_fs: AbstractFileSystem,
    dest_fs: AbstractFileSystem,
    source_path: str,
    dest_path: str,
    ignore_patterns: list[str] | None = None,
) -> list[tuple[str, str, str]]:
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
            with source_fs.open(src, "rb") as source_file:
                with dest_fs.open(dst, "wb") as dest_file:
                    dest_file.write(source_file.read())
            return "Success"
        except Exception as e:
            logger.error(f"Failed to copy {src} to {dst}: {e!s}")
            return f"Failed: {e!s}"

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


# Example usage
def copy_data(
    source_config: StorageConfig,
    dest_config: StorageConfig,
    source_path: str,
    dest_path: str,
    ignore_patterns: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    source_fs = _create_filesystem(source_config)
    dest_fs = _create_filesystem(dest_config)

    return copy_between_filesystems(source_fs, dest_fs, source_path, dest_path, ignore_patterns)
