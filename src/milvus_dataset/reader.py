import pandas as pd
import pyarrow.parquet as pq
from typing import Optional, Generator, Any
from .logging import logger


class DatasetReader:
    def __init__(self, dataset):
        self.dataset = dataset

    def read(self, mode: str = 'stream', batch_size: Optional[int] = None) -> pd.DataFrame | Generator[
        Any, Any, None] | Any:
        path = f"{self.dataset.root_path}/{self.dataset.name}/{self.dataset.split}"
        logger.info(f"Attempting to read dataset from path: {path}")

        try:
            # 检查路径是否存在
            if not self.dataset.fs.exists(path):
                logger.warning(f"Dataset path does not exist: {path}")


            # 列出目录内容以进行调试
            try:
                contents = self.dataset.fs.ls(path)
                logger.info(f"Contents of {path}: {contents}")
            except Exception as e:
                logger.warning(f"Unable to list contents of {path}: {str(e)}")

            if mode == 'full':
                return self._read_full(path)
            elif mode == 'stream':
                return self._read_stream(path, 1)
            elif mode == 'batch':
                if batch_size is None:
                    raise ValueError("Batch size must be provided when using 'batch' read mode.")
                return self._read_stream(path, batch_size)
            else:
                raise ValueError("Invalid read mode. Expected 'stream', 'batch', or 'full'.")

        except Exception as e:
            logger.error(f"Error reading dataset: {str(e)}")
            raise

    def _read_full(self, path):
        if self.dataset.fs.isfile(path):
            with self.dataset.fs.open(path, 'rb') as f:
                return pq.read_table(f).to_pandas()
        else:
            logger.info(f"Reading full dataset from: {path}, this may take a while...")
            file_list = self.dataset.fs.glob(f"{path}/*.parquet")
            if not file_list:
                logger.info("No parquet files found.")
                return pd.DataFrame()
            else:
                logger.info(f"Found {len(file_list)} parquet files.")
                dfs = []
                for file in file_list:
                    with self.dataset.fs.open(file, 'rb') as f:
                        dfs.append(pq.read_table(f).to_pandas())
                return pd.concat(dfs, ignore_index=True)

    def _read_stream(self, path, batch_size):
        def data_generator():
            current_batch = []
            current_size = 0

            for file in self.dataset.fs.glob(f"{path}/*.parquet"):
                with self.dataset.fs.open(file, 'rb') as f:
                    pf = pq.ParquetFile(f)
                    for batch in pf.iter_batches():
                        df = batch.to_pandas()
                        current_batch.append(df)
                        current_size += len(df)

                        while current_size >= batch_size:
                            # Combine and yield a full batch
                            combined_df = pd.concat(current_batch, ignore_index=True)

                            batch_df = combined_df.iloc[:batch_size]
                            logger.info(f"Yielding batch of size {len(batch_df)}")
                            yield batch_df

                            # Keep the remainder for the next batch
                            if len(combined_df) > batch_size:
                                current_batch = [combined_df.iloc[batch_size:]]
                                current_size = len(current_batch[0])
                            else:
                                current_batch = []
                                current_size = 0

            # Yield any remaining data
            if current_batch:
                yield pd.concat(current_batch, ignore_index=True)

        return data_generator()
