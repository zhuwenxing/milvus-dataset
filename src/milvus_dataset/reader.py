import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from typing import Optional, Iterator, Dict, Any, Union, Generator
from pathlib import Path

from pandas import DataFrame
from .logging import logger

class ArrowBasedDataset:
    def __init__(self, data_source, mode: str = 'full', batch_size: Optional[int] = None):
        self.data_source = data_source
        self.mode = mode
        self.batch_size = batch_size if mode != 'stream' else 1
        self._length = 0
        self._features = None
        self._pf_list = []

        if os.path.isdir(data_source):
            self._file_list = [os.path.join(data_source, f) for f in os.listdir(data_source) if f.endswith('.parquet')]
        else:
            self._file_list = [data_source]

        if mode == 'full':
            self._table = pa.concat_tables([pq.read_table(f) for f in self._file_list])
            self._length = len(self._table)
            self._features = self._infer_features(self._table.schema)
        else:
            self._table = None
            for file in self._file_list:
                pf = pq.ParquetFile(file)
                self._pf_list.append(pf)
                self._length += pf.metadata.num_rows
            if self._pf_list:
                self._features = self._infer_features(self._pf_list[0].schema_arrow)

    def _infer_features(self, schema):
        return {name: pa.struct([pa.field(name, field.type)])
                for name, field in zip(schema.names, schema)}

    def __len__(self):
        return self._length

    @property
    def features(self):
        return self._features

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.mode == 'full':
            return {name: self._table[name][index].as_py() for name in self._table.column_names}
        else:
            # Find the file and row group containing the specified row
            rows_seen = 0
            for pf in self._pf_list:
                if rows_seen + pf.metadata.num_rows > index:
                    break
                rows_seen += pf.metadata.num_rows

            row_group_index = 0
            while rows_seen + pf.metadata.row_group(row_group_index).num_rows <= index:
                rows_seen += pf.metadata.row_group(row_group_index).num_rows
                row_group_index += 1

            # Read the row group containing the specified row
            row_group = pf.read_row_group(row_group_index)

            # Calculate the index of the row within the row group
            row_in_group = index - rows_seen

            # Return the data for the specified row
            return {name: row_group[name][row_in_group].as_py() for name in row_group.column_names}

    def __iter__(self) -> Iterator[Union[Dict[str, Any], pd.DataFrame]]:
        if self.mode == 'full':
            yield self._table.to_pandas()
        else:
            for pf in self._pf_list:
                yield from (batch.to_pandas() for batch in pf.iter_batches(batch_size=self.batch_size))

    def to_pandas(self) -> pd.DataFrame:
        if self.mode == 'full':
            return self._table.to_pandas()
        else:
            return pd.concat([batch for batch in self])

    def __del__(self):
        for pf in self._pf_list:
            pf.close()

class DatasetReader:
    def __init__(self, dataset):
        self.dataset = dataset

    def read(self, mode: str = 'stream', batch_size: Optional[int] = None) -> DataFrame | Generator[
        Any, Any, None] | Any:
        path = Path(self.dataset.root_path) / self.dataset.name / self.dataset.split
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if mode == 'full':
            if path.is_file():
                return pq.read_table(str(path)).to_pandas()
            else:
                logger.info(f"Reading full dataset from: {path}, this may take a while...")
                file_list = list(path.glob('*.parquet'))
                if not file_list:
                    logger.info("No parquet files found.")
                    return pd.DataFrame()
                else:
                    logger.info(f"Found {len(file_list)} parquet files.")
                    return pd.concat([pq.read_table(str(f)).to_pandas() for f in path.glob('*.parquet')], ignore_index=True)
        elif mode in ['stream', 'batch']:
            def data_generator():
                for file in path.glob('*.parquet'):
                    pf = pq.ParquetFile(str(file))
                    for batch in pf.iter_batches(batch_size=batch_size):
                        yield batch.to_pandas()
            return data_generator()

        else:
            raise ValueError("Invalid read mode. Expected 'stream', 'batch', or 'full'.")
