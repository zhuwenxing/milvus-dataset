import math

import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from typing import Union, Dict, List
import pyarrow as pa
import tempfile
import pyarrow.parquet as pq
import glob
from .logging import logger


class DatasetWriter:
    def __init__(self, dataset, target_file_size_mb=5, tolerance=0.1):
        self.dataset = dataset
        self.target_file_size_bytes = target_file_size_mb * 1024 * 1024
        self.tolerance = tolerance  # 允许的文件大小偏差

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

    def _estimate_parquet_size(self, df: pd.DataFrame) -> int:
        with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
            df.to_parquet(tmp.name, engine='pyarrow', compression='snappy')
            return Path(tmp.name).stat().st_size

    def _write_table(self, table: pa.Table, mode: str):
        base_path = Path(f"{self.dataset.root_path}/{self.dataset.name}/{self.dataset.split}")
        base_path.mkdir(parents=True, exist_ok=True)

        df = table.to_pandas()
        ddf = dd.from_pandas(df, chunksize=100000)  # 使用较大的 chunksize 来减少开销

        # 估计 Parquet 文件大小
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
        estimated_row_size = self._estimate_parquet_size(sample_df) / len(sample_df)
        estimated_total_size = estimated_row_size * len(df)

        logger.info(f"Estimated total Parquet size: {estimated_total_size / (1024 * 1024):.2f} MB")

        # 计算分区数
        num_partitions = max(1, math.ceil(estimated_total_size / self.target_file_size_bytes))
        rows_per_partition = math.ceil(len(df) / num_partitions)

        logger.info(
            f"Writing data with {num_partitions} partitions, approximately {rows_per_partition} rows per partition")

        start_number = 0
        if mode == 'append':
            existing_files = sorted(glob.glob(str(base_path / "part-*.parquet")))
            if existing_files:
                last_file = existing_files[-1]
                start_number = int(last_file.split('-')[-1].split('.')[0]) + 1

        def get_filename(partition_index):
            file_number = start_number + partition_index
            return f"part-{file_number:05d}.parquet"

        # 使用 Dask 的 to_parquet 方法，但自定义分区策略
        ddf.repartition(npartitions=num_partitions).to_parquet(
            base_path,
            engine='pyarrow',
            compression='snappy',
            write_metadata_file=False,
            name_function=get_filename
        )

        # 检查并调整文件大小
        self._adjust_file_sizes(base_path)

        # 更新元数据
        self.dataset.metadata['last_file_number'] = start_number + num_partitions - 1
        self.dataset._save_metadata()

    def _adjust_file_sizes(self, base_path):
        files = sorted(glob.glob(str(base_path / "part-*.parquet")))
        logger.info(f"Found {len(files)} files for size adjustment")
        for file in files:
            file_path = Path(file)
            if not file_path.exists():
                logger.warning(f"File not found during size adjustment: {file}")
                continue
            try:
                file_size = file_path.stat().st_size
                logger.debug(f"Checking file: {file}, size: {file_size / (1024*1024):.2f} MB")
                if file_size < self.target_file_size_bytes * (1 - self.tolerance):
                    # 如果文件太小，尝试与下一个文件合并
                    next_file = next((f for f in files if f > file), None)
                    if next_file:
                        self._merge_files(file, next_file)
                        logger.info(f"Merged {file} with {next_file}")
                elif file_size > self.target_file_size_bytes * (1 + self.tolerance):
                    # 如果文件太大，尝试拆分
                    self._split_file(file)
                    logger.info(f"Split file: {file}")
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")

    def _merge_files(self, file1, file2):
        df1 = pd.read_parquet(file1)
        df2 = pd.read_parquet(file2)
        merged_df = pd.concat([df1, df2])
        merged_df.to_parquet(file1, engine='pyarrow', compression='snappy')
        Path(file2).unlink()

    def _split_file(self, file):
        df = pd.read_parquet(file)
        mid = len(df) // 2
        df1 = df.iloc[:mid]
        df2 = df.iloc[mid:]
        df1.to_parquet(file, engine='pyarrow', compression='snappy')
        new_file = str(Path(file).parent / f"{Path(file).stem}_split.parquet")
        df2.to_parquet(new_file, engine='pyarrow', compression='snappy')

    def _get_chunk(self, table: pa.Table) -> pa.Table:
        # Estimate number of rows for target file size
        row_size = table.nbytes / len(table)
        target_rows = int(self.target_file_size_bytes / row_size)
        return table.slice(0, min(target_rows, len(table)))
