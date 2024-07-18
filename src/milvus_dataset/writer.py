import math
import pandas as pd
import pyarrow as pa
from pathlib import Path
from typing import Union, Dict, List
import tempfile
import glob
from .logging import logger


class DatasetWriter:
    def __init__(self, dataset, target_file_size_mb=5):
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

        with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
            sample_df.to_parquet(tmp.name, engine='pyarrow', compression='snappy')
            file_size = Path(tmp.name).stat().st_size

        estimated_row_size = file_size / len(sample_df)
        estimated_rows_per_file = int(self.target_file_size_bytes / estimated_row_size)

        # Round to nearest 10000
        rounded_rows_per_file = round(estimated_rows_per_file / 10000) * 10000

        # Ensure we always have at least 10000 rows per file
        final_rows_per_file = max(10000, rounded_rows_per_file)

        logger.info(f"Estimated rows per file: {estimated_rows_per_file}")
        logger.info(f"Rounded rows per file: {final_rows_per_file}")
        return final_rows_per_file

    def _write_table(self, table: pa.Table, mode: str):
        base_path = Path(f"{self.dataset.root_path}/{self.dataset.name}/{self.dataset.split}")
        base_path.mkdir(parents=True, exist_ok=True)

        df = table.to_pandas()

        if self.rows_per_file is None:
            self.rows_per_file = self._estimate_rows_per_file(df)

        start_number = 0
        if mode == 'append':
            existing_files = sorted(glob.glob(str(base_path / "part-*.parquet")))
            if existing_files:
                last_file = existing_files[-1]
                start_number = int(last_file.split('-')[-1].split('.')[0]) + 1
                # Read the last file to check if it's full
                last_df = pd.read_parquet(last_file)
                if len(last_df) < self.rows_per_file:
                    # If the last file is not full, we'll append to it
                    start_number -= 1
                    df = pd.concat([last_df, df])

        num_full_files = len(df) // self.rows_per_file
        for i in range(num_full_files):
            start_idx = i * self.rows_per_file
            end_idx = (i + 1) * self.rows_per_file
            partition_df = df.iloc[start_idx:end_idx]

            filename = f"part-{start_number + i:05d}.parquet"
            file_path = base_path / filename
            partition_df.to_parquet(file_path, engine='pyarrow', compression='snappy')
            logger.info(f"Wrote file: {filename} with {len(partition_df)} rows")

        # Handle the last, potentially partial file
        if len(df) % self.rows_per_file > 0:
            start_idx = num_full_files * self.rows_per_file
            partition_df = df.iloc[start_idx:]
            filename = f"part-{start_number + num_full_files:05d}.parquet"
            file_path = base_path / filename
            partition_df.to_parquet(file_path, engine='pyarrow', compression='snappy')
            logger.info(f"Wrote last file: {filename} with {len(partition_df)} rows")

        # Log detailed information about all files
        logger.info("Summary of all written files:")
        for file_path in sorted(base_path.glob("part-*.parquet")):
            file_size = file_path.stat().st_size
            df = pd.read_parquet(file_path)
            logger.info(f"File: {file_path.name}, Size: {file_size / 1024:.2f} KB, Rows: {len(df)}")

        # 更新元数据
        self.dataset.metadata['last_file_number'] = start_number + num_full_files + (
            1 if len(df) % self.rows_per_file > 0 else 0) - 1
        self.dataset._save_metadata()
