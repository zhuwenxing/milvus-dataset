import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Union, Dict, List
import tempfile
from .log_config import logger

class DatasetWriter:
    def __init__(self, dataset, target_file_size_mb=512):
        self.dataset = dataset
        self.target_file_size_bytes = target_file_size_mb * 1024 * 1024
        self.rows_per_file = None

    def write(self, data: Union[pd.DataFrame, Dict, List[Dict]], mode: str = 'append'):
        if isinstance(data, pd.DataFrame):
            return self._write_dataframe(data, mode)
        elif isinstance(data, dict):
            return self._write_dict(data, mode)
        elif isinstance(data, list):
            return self._write_list(data, mode)
        else:
            raise ValueError("Unsupported data type. Expected DataFrame, Dict, or List[Dict].")

    def _write_dataframe(self, df: pd.DataFrame, mode: str):
        table = pa.Table.from_pandas(df)
        self._write_table(table, mode)

    def _write_dict(self, data: Dict, mode: str):
        df = pd.DataFrame(data)
        return self._write_dataframe(df, mode)

    def _write_list(self, data: List[Dict], mode: str):
        df = pd.DataFrame(data)
        return self._write_dataframe(df, mode)

    def _estimate_rows_per_file(self, df: pd.DataFrame) -> int:
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size) if len(df) > sample_size else df

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            sample_df.to_parquet(tmp.name, engine='pyarrow', compression='snappy')

            # 使用 os.path.getsize 获取本地文件大小
            file_size = os.path.getsize(tmp.name)

        # 删除临时文件
        os.unlink(tmp.name)

        estimated_row_size = file_size / len(sample_df)
        estimated_rows_per_file = int(self.target_file_size_bytes / estimated_row_size)

        # 四舍五入到最接近的 10000
        rounded_rows_per_file = round(estimated_rows_per_file / 10000) * 10000

        # 确保每个文件至少有 10000 行
        final_rows_per_file = max(10000, rounded_rows_per_file)

        logger.info(f"Estimated rows per file: {estimated_rows_per_file}")
        logger.info(f"Rounded rows per file: {final_rows_per_file}")
        return final_rows_per_file

    def _write_table(self, table: pa.Table, mode: str):
        base_path = f"{self.dataset.root_path}/{self.dataset.name}/{self.dataset.split}"
        self.dataset.fs.makedirs(base_path, exist_ok=True)

        df = table.to_pandas()

        if self.rows_per_file is None:
            self.rows_per_file = self._estimate_rows_per_file(df)

        start_number = 0
        if mode == 'append':
            existing_files = sorted(self.dataset.fs.glob(f"{base_path}/part-*.parquet"))
            if existing_files:
                last_file = existing_files[-1]
                start_number = int(last_file.split('-')[-1].split('.')[0]) + 1
                with self.dataset.fs.open(last_file, 'rb') as f:
                    last_df = pd.read_parquet(f)
                if len(last_df) < self.rows_per_file:
                    start_number -= 1
                    df = pd.concat([last_df, df])

        num_full_files = len(df) // self.rows_per_file
        for i in range(num_full_files):
            start_idx = i * self.rows_per_file
            end_idx = (i + 1) * self.rows_per_file
            partition_df = df.iloc[start_idx:end_idx]

            filename = f"part-{start_number + i:05d}.parquet"
            file_path = f"{base_path}/{filename}"
            with self.dataset.fs.open(file_path, 'wb') as f:
                partition_df.to_parquet(f, engine='pyarrow', compression='snappy')
            logger.info(f"Wrote file: {filename} with {len(partition_df)} rows")

        if len(df) % self.rows_per_file > 0:
            start_idx = num_full_files * self.rows_per_file
            partition_df = df.iloc[start_idx:]
            filename = f"part-{start_number + num_full_files:05d}.parquet"
            file_path = f"{base_path}/{filename}"
            with self.dataset.fs.open(file_path, 'wb') as f:
                partition_df.to_parquet(f, engine='pyarrow', compression='snappy')
            logger.info(f"Wrote last file: {filename} with {len(partition_df)} rows")

        self.dataset.metadata['last_file_number'] = start_number + num_full_files + (
            1 if len(df) % self.rows_per_file > 0 else 0) - 1
        self.dataset._save_metadata()
