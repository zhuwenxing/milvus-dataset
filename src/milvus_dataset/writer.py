import pandas as pd
from typing import Union, Dict, List
import pyarrow as pa
import pyarrow.parquet as pq

class DatasetWriter:
    def __init__(self, dataset):
        self.dataset = dataset

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
        df = pd.DataFrame([data])
        return self._write_dataframe(df, mode)

    def _write_list(self, data: List[Dict], mode: str):
        df = pd.DataFrame(data)
        return self._write_dataframe(df, mode)

    def _write_table(self, table: pa.Table, mode: str):
        path = self.dataset.storage.join(self.dataset.root_path, f"{self.dataset.name}.parquet")
        if mode == 'append' and self.dataset.storage.exists(path):
            existing_table = pq.read_table(path)
            table = pa.concat_tables([existing_table, table])
        pq.write_table(table, path)

