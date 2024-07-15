import pandas as pd
import pyarrow.parquet as pq

class DatasetReader:
    def __init__(self, dataset):
        self.dataset = dataset

    def read(self, mode: str = 'stream', batch_size: int = 1000):
        path = self.dataset.storage.join(self.dataset.root_path, f"{self.dataset.name}.parquet")
        if not self.dataset.storage.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        if mode == 'stream':
            return self._read_stream(path)
        elif mode == 'batch':
            return self._read_batch(path, batch_size)
        elif mode == 'full':
            return self._read_full(path)
        else:
            raise ValueError("Invalid read mode. Expected 'stream', 'batch', or 'full'.")

    def _read_stream(self, path):
        for batch in pq.ParquetFile(path).iter_batches():
            yield batch.to_pandas()

    def _read_batch(self, path, batch_size):
        for batch in pq.read_table(path, batch_size=batch_size).to_batches():
            yield batch.to_pandas()

    def _read_full(self, path):
        return pq.read_table(path).to_pandas()
