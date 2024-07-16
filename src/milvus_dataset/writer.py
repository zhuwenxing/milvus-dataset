import pandas as pd
from pathlib import Path
from typing import Union, Dict, List
import pyarrow as pa
import pyarrow.parquet as pq
import glob
class DatasetWriter:
    def __init__(self, dataset, target_file_size_mb=512):
        self.dataset = dataset
        self.target_file_size_bytes = target_file_size_mb * 1024 * 1024

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

    def _write_table(self, table: pa.Table, mode: str):
        base_path = Path(f"{self.dataset.root_path}/{self.dataset.name}")
        base_path.mkdir(parents=True, exist_ok=True)

        if mode == 'append':
            existing_files = sorted(glob.glob(str(base_path / "part-*.parquet")))
            if existing_files:
                last_file = existing_files[-1]
                last_number = int(last_file.split('-')[-1].split('.')[0])
                start_number = last_number + 1

                # Check if the last file can accommodate more data
                last_file_size = Path(last_file).stat().st_size
                if last_file_size < self.target_file_size_bytes:
                    last_table = pq.read_table(last_file)
                    combined_table = pa.concat_tables([last_table, table])
                    pq.write_table(combined_table,last_file, compression='snappy')
                    if Path(last_file).stat().st_size >= self.target_file_size_bytes:
                        # If the file is now full, start a new file with remaining data
                        table = combined_table.slice(len(last_table))
                    else:
                        return  # All data has been written
            else:
                start_number = 0
        else:
            # In overwrite mode, start from 0
            start_number = 0

        # Write remaining data to new files
        file_number = start_number
        while len(table) > 0:
            file_path = base_path / f"part-{file_number:05d}.parquet"
            chunk = self._get_chunk(table)
            pq.write_table(chunk, str(file_path), compression='snappy')
            table = table.slice(len(chunk))
            file_number += 1

    def _get_chunk(self, table: pa.Table) -> pa.Table:
        # Estimate number of rows for target file size
        row_size = table.nbytes / len(table)
        target_rows = int(self.target_file_size_bytes / row_size)
        return table.slice(0, min(target_rows, len(table)))

