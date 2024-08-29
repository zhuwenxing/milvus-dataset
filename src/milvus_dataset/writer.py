import os
import time
import threading
import uuid
from queue import Queue
from typing import Union, Dict, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
from .log_config import logger

class DatasetWriter:
    def __init__(self, dataset, target_file_size_mb=512, num_buffers=4, queue_size=10):
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
            except Exception as e:
                break
        logger.info(f"items in queue: {items}")
        self.write_threads = []
        self.current_buffer = 0
        self.file_counter = 0
        self.mode = 'append'

    def __enter__(self):
        self._start_write_threads()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._flush_all_buffers()
        self._stop_write_threads()

    def _start_write_threads(self):
        for _ in range(self.num_buffers):
            thread = threading.Thread(target=self._write_worker)
            thread.daemon = True
            thread.start()
            self.write_threads.append(thread)

    def _stop_write_threads(self):
        for _ in range(self.num_buffers):
            self.write_queue.put(None)
        for thread in self.write_threads:
            thread.join()

    def _write_worker(self):
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
            logger.info(f"Writing buffer {buffer_df}, {self.write_queue.qsize()}")
            if not isinstance(buffer_df, pd.DataFrame):
                logger.error(f"Invalid buffer type: {type(buffer_df)}")
                self.write_queue.task_done()
                continue
            self._write_buffer(buffer_df)
            self.write_queue.task_done()
            time.sleep(0.1)

    def write(self, data: Union[pd.DataFrame, Dict, List[Dict]], mode: str = 'append', verify_schema: bool = True):
        self.mode = mode
        if verify_schema:
            self.dataset._verify_schema(data)
        if isinstance(data, pd.DataFrame):
            self._write_dataframe(data)
        elif isinstance(data, dict):
            self._write_dict(data)
        elif isinstance(data, list):
            self._write_list(data)
        else:
            raise ValueError("Unsupported data type. Expected DataFrame, Dict, or List[Dict].")
        # self.dataset.summary()

    def _write_dataframe(self, df: pd.DataFrame):
        # logger.info(f"Writing {len(df)} rows to dataset")
        if self.rows_per_file is None:
            self.rows_per_file = self._estimate_rows_per_file(df)
        t0 = time.time()
        batch_size = 10000
        for batch in range(0, len(df), batch_size):
            df_batch = df[batch:batch + batch_size]

            with self.buffer_locks[self.current_buffer]:
                # logger.info(f"Adding {len(df_batch)} row to buffer {self.current_buffer}")
                self.buffers[self.current_buffer].extend(df_batch.to_dict(orient='records'))
                # logger.info(f"buffer {len(self.buffers[self.current_buffer])} rows")
                if len(self.buffers[self.current_buffer]) >= self.rows_per_file:
                    logger.info(f"Buffer {self.current_buffer} is full, len {len(self.buffers[self.current_buffer])}, adding to write queue")
                    df = pd.DataFrame(self.buffers[self.current_buffer])
                    while True:
                        try:
                            logger.debug(f"Adding buffer {self.current_buffer} to write queue, data size {len(df)}")
                            if not self.write_queue.full():
                                self.write_queue.put(df)
                                break
                        except Exception as e:
                            if self.write_queue.full():
                                logger.warning("Write queue is full. Waiting for space...")
                            logger.warning(f"Write queue is failed with error {e}")
                    logger.info(f"current write queue size: {self.write_queue.qsize()}")
                    self.buffers[self.current_buffer] = []
                    self.current_buffer = (self.current_buffer + 1) % self.num_buffers
        tt = time.time() - t0
        # logger.info(f"Data writing time: {tt:.2f} s")

    def _write_dict(self, data: Dict):
        df = pd.DataFrame(data)
        self._write_dataframe(df)

    def _write_list(self, data: List[Dict]):
        df = pd.DataFrame(data)
        self._write_dataframe(df)

    def _estimate_rows_per_file(self, df: pd.DataFrame) -> int:
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size) if len(df) > sample_size else df

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            sample_df.to_parquet(tmp.name, engine='pyarrow', compression='snappy')
            file_size = os.path.getsize(tmp.name)

        os.unlink(tmp.name)

        estimated_row_size = file_size / len(sample_df)
        estimated_rows_per_file = int(self.target_file_size_bytes / estimated_row_size)
        rounded_rows_per_file = round(estimated_rows_per_file / 10000) * 10000
        final_rows_per_file = max(10000, rounded_rows_per_file)

        logger.info(f"Estimated rows per file: {estimated_rows_per_file}")
        logger.info(f"Rounded rows per file: {final_rows_per_file}")
        return final_rows_per_file

    def _write_buffer(self, buffer_df: pd.DataFrame):
        df = buffer_df
        base_path = f"{self.dataset.root_path}/{self.dataset.name}/{self.dataset.split}"
        self.dataset.fs.makedirs(base_path, exist_ok=True)

        if self.mode == 'append' and self.file_counter == 0:
            existing_files = sorted(self.dataset.fs.glob(f"{base_path}/part-*.parquet"))
            if existing_files:
                last_file = existing_files[-1]
                self.file_counter = int(last_file.split('-')[-1].split('.')[0]) + 1

        filename = f"{uuid.uuid4()}.parquet"
        file_path = f"{base_path}/{filename}"

        with self.dataset.fs.open(file_path, 'wb') as f:
            logger.info(f"Writing {len(df)} rows to file: {file_path}, buffer {df}")
            df.to_parquet(f, engine='pyarrow', compression='snappy')

        logger.info(f"Wrote file: {filename} with {len(df)} rows")
        self.file_counter += 1

    def _flush_all_buffers(self):
        for i in range(self.num_buffers):
            if self.buffers[i]:
                self.write_queue.put(pd.DataFrame(self.buffers[i]))
        self.write_queue.join()

    def _save_metadata(self):
        self.dataset.metadata['last_file_number'] = self.file_counter - 1
        self.dataset._save_metadata()
