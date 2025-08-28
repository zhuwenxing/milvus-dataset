"""
Writer module for efficient dataset writing and management.

This module provides functionality for writing large datasets efficiently,
with features like automatic file size management, buffered writing,
and concurrent I/O operations.
"""

__all__ = ["DatasetWriter"]

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from queue import Queue
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .core import Dataset


from .log_config import logger


class DatasetWriter:
    """A class for efficiently writing data to a dataset.

    This class implements a buffered writing system with concurrent I/O operations
    to efficiently handle large-scale data writing tasks. It automatically manages
    file sizes and provides mechanisms for data buffering and batch processing.

    Args:
        dataset (Dataset): The target dataset to write to
        target_file_size_mb (int): Target size for individual files in megabytes (default: 512)
        num_buffers (int): Number of buffer workers for concurrent writing (default: 10)
        queue_size (int): Size of the queue for buffering data (default: 20)
    """

    def __init__(
        self,
        dataset: "Dataset",
        target_file_size_mb: int = 512,
        num_buffers: int = 10,
        queue_size: int = 20,
    ) -> None:
        self.dataset = dataset
        self.target_file_size_bytes = target_file_size_mb * 1024 * 1024
        self.rows_per_file = None
        self.num_buffers = num_buffers
        self.buffers = [[] for _ in range(num_buffers)]
        self.buffer_locks = [threading.Lock() for _ in range(num_buffers)]
        self.write_queue = Queue(maxsize=queue_size)
        logger.info(f"queue length: {self.write_queue.qsize()}")
        items = []
        while not self.write_queue.empty():
            try:
                items.append(self.write_queue.get(block=False))
            except Exception:
                break
        logger.debug(f"Queue status: size={self.write_queue.qsize()}, items={items}")
        self.write_threads = []
        self.current_buffer = 0
        self.file_counter = 0
        self.file_counter_lock = threading.Lock()  # 线程安全的计数器
        self.mode = "append"

    def __enter__(self) -> "DatasetWriter":
        self._start_write_threads()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._flush_all_buffers()
        self._stop_write_threads()

        # Update metadata timestamp if it exists
        metadata_file = f"{self.dataset.root_path}/{self.dataset.name}/metadata.json"
        if self.dataset.fs.exists(metadata_file):
            try:
                with self.dataset.fs.open(metadata_file, "r") as f:
                    metadata = json.load(f)
                metadata["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
                logger.info(f"metadata_file: {metadata_file}")
                with self.dataset.fs.open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                    logger.info(f"Updated metadata timestamp: {metadata['updated_at']}")
                logger.info(f"Updated metadata {metadata}")
            except Exception as e:
                logger.error(f"Error updating metadata timestamp: {e}")
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
        with self.dataset.fs.open(metadata_file, "r") as f:
            metadata = json.load(f)
        logger.info(f"metadata: {metadata}")

    def _start_write_threads(self) -> None:
        for _ in range(self.num_buffers):
            thread = threading.Thread(target=self._write_worker)
            thread.daemon = True
            thread.start()
            self.write_threads.append(thread)

    def _stop_write_threads(self) -> None:
        for _ in range(self.num_buffers):
            self.write_queue.put(None)
        for thread in self.write_threads:
            thread.join()

    def _write_worker(self) -> None:
        while True:
            logger.info(f"write queue size: {self.write_queue.qsize()}")
            item = self.write_queue.get()
            # items = []
            # while not self.write_queue.empty():
            #     try:
            #         items.append(self.write_queue.get(block=False))
            #     except Exception as e:
            #         break
            # logger.info(f"get queue item: {items}")
            if item is None:
                break
            buffer_df = item
            try:
                logger.debug(
                    f"Processing buffer: size={len(buffer_df)}, queue_size={self.write_queue.qsize()}"
                )

                if not isinstance(buffer_df, pd.DataFrame):
                    logger.error(
                        f"Invalid buffer type: expected=DataFrame, actual={type(buffer_df)}"
                    )
                    raise ValueError(f"Invalid buffer type: {type(buffer_df)}")
            except Exception as e:
                logger.exception(f"Error processing buffer: {e}")
            self._write_buffer(buffer_df)
            self.write_queue.task_done()
            time.sleep(0.1)

    def write(
        self,
        data: pd.DataFrame | dict | list[dict],
        mode: str = "append",
        verify_schema: bool = True,
    ) -> None:
        self.mode = mode

        if isinstance(data, pd.DataFrame):
            self._write_dataframe(data)
        elif isinstance(data, dict):
            self._write_dict(data)
        elif isinstance(data, list):
            self._write_list(data)
        else:
            raise ValueError("Unsupported data type. Expected DataFrame, Dict, or List[Dict].")
        # self.dataset.summary()

    def _write_dataframe(self, df: pd.DataFrame) -> None:
        # logger.info(f"Writing {len(df)} rows to dataset")
        self.dataset.verify_schema(df)
        if self.rows_per_file is None:
            self.rows_per_file = self._estimate_rows_per_file(df)
        batch_size = 10000
        df_chunks = [df[batch : batch + batch_size] for batch in range(0, len(df), batch_size)]
        for i, df_batch in enumerate(df_chunks):
            logger.debug(f"Processing batch {i+1}/{len(df_chunks)}: size={len(df_batch)}")

            with self.buffer_locks[self.current_buffer]:
                # logger.info(f"Adding {len(df_batch)} row to buffer {self.current_buffer}")
                self.buffers[self.current_buffer].extend(df_batch.to_dict(orient="records"))
                # logger.info(f"buffer {len(self.buffers[self.current_buffer])} rows")
                if len(self.buffers[self.current_buffer]) >= self.rows_per_file:
                    logger.info(
                        f"Buffer {self.current_buffer} is full, len {len(self.buffers[self.current_buffer])}, adding to write queue"
                    )
                    df = pd.DataFrame(self.buffers[self.current_buffer])
                    try:
                        logger.debug(
                            f"Adding buffer {self.current_buffer} to write queue, data size {len(df)}"
                        )
                        if not self.write_queue.full():
                            self.write_queue.put(df)
                    except Exception as e:
                        if self.write_queue.full():
                            logger.warning("Write queue is full. Waiting for space...")
                        logger.warning(f"Write queue is failed with error {e}")
                    logger.debug(
                        f"Queue status after write: size={len(self.buffers[self.current_buffer])}"
                    )
                    self.buffers[self.current_buffer] = []
                    self.current_buffer = (self.current_buffer + 1) % self.num_buffers

    def _write_dict(self, data: dict) -> None:
        df = pd.DataFrame(data)
        self._write_dataframe(df)

    def _write_list(self, data: list[dict]) -> None:
        df = pd.DataFrame(data)
        self._write_dataframe(df)

    def _estimate_rows_per_file(self, df: pd.DataFrame) -> int:
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size) if len(df) > sample_size else df

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            sample_df.to_parquet(tmp.name, engine="pyarrow", compression="snappy")
            file_size = os.path.getsize(tmp.name)

        os.unlink(tmp.name)

        estimated_row_size = file_size / len(sample_df)
        estimated_rows_per_file = int(self.target_file_size_bytes / estimated_row_size)
        rounded_rows_per_file = round(estimated_rows_per_file / 10000) * 10000
        final_rows_per_file = max(10000, rounded_rows_per_file)

        logger.info(
            f"Calculated file parameters: estimated_rows={estimated_rows_per_file}, final_rows={final_rows_per_file}"
        )
        return final_rows_per_file

    def _write_buffer(self, buffer_df: pd.DataFrame) -> None:
        df = buffer_df
        base_path = f"{self.dataset.root_path}/{self.dataset.name}/{self.dataset.split}"
        self.dataset.fs.makedirs(base_path, exist_ok=True)

        if self.mode == "append" and self.file_counter == 0:
            existing_files = sorted(
                self.dataset.fs.glob(f"{base_path}/{self.dataset.split}-part-*.parquet")
            )
            if existing_files:
                last_file = existing_files[-1]
                self.file_counter = int(last_file.split("-")[-1].split(".")[0])

        # 线程安全的文件计数
        with self.file_counter_lock:
            self.file_counter += 1
            current_counter = self.file_counter

        filename = f"{self.dataset.split}-part-{current_counter:06d}.parquet"
        file_path = f"{base_path}/{filename}"

        try:
            logger.info(f"Writing data to file: path={file_path}, rows={len(df)}")
            with self.dataset.fs.open(file_path, "wb") as f:
                df.to_parquet(f, engine="pyarrow", compression="snappy")
            logger.info(f"Successfully wrote file: name={filename}, rows={len(df)}")
        except Exception as e:
            logger.exception(f"Failed to write file: name={filename}, error={e}")
            raise

    def _flush_all_buffers(self) -> None:
        for i in range(self.num_buffers):
            if self.buffers[i]:
                self.write_queue.put(pd.DataFrame(self.buffers[i]))
        self.write_queue.join()

    def _save_metadata(self) -> None:
        self.dataset.metadata["last_file_number"] = self.file_counter - 1
        self.dataset._save_metadata()
