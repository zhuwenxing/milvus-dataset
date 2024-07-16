import unittest
from unittest.mock import Mock, MagicMock, patch
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import tempfile
import os

# 假设 DatasetReader 和 ArrowBasedDataset 在 your_module 中定义
from milvus_dataset.reader import DatasetReader, ArrowBasedDataset


class TestDatasetReader(unittest.TestCase):
    def setUp(self):
        # 创建一个临时 parquet 文件用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'test.parquet')

        # 创建一个简单的 DataFrame 并保存为 parquet
        df = pd.DataFrame({
            'id': range(100),
            'value': [f'value_{i}' for i in range(100)]
        })
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.temp_file)

        # 模拟 dataset 对象
        self.mock_dataset = Mock()
        self.mock_dataset.storage.join.return_value = self.temp_file
        self.mock_dataset.storage.exists.return_value = True
        self.mock_dataset.root_path = self.temp_dir
        self.mock_dataset.name = 'test'

        # 创建 DatasetReader 实例
        self.reader = DatasetReader(self.mock_dataset)

    def tearDown(self):
        # 清理临时文件
        os.remove(self.temp_file)
        os.rmdir(self.temp_dir)

    def test_read_stream_mode(self):
        dataset = self.reader.read(mode='stream')
        self.assertIsInstance(dataset, ArrowBasedDataset)
        self.assertEqual(dataset.mode, 'stream')
        self.assertEqual(dataset.batch_size, 1)

        # 测试迭代
        count = sum(1 for _ in dataset)
        self.assertEqual(count, 100)  # 应该有 100 行数据

    def test_read_batch_mode(self):
        batch_size = 10
        dataset = self.reader.read(mode='batch', batch_size=batch_size)
        self.assertIsInstance(dataset, ArrowBasedDataset)
        self.assertEqual(dataset.mode, 'batch')
        self.assertEqual(dataset.batch_size, batch_size)

        # 测试批量迭代
        batches = list(dataset)
        self.assertEqual(len(batches), 10)  # 应该有 10 个批次
        self.assertEqual(len(batches[0]), batch_size)  # 每个批次应该有 10 行

    def test_read_full_mode(self):
        dataset = self.reader.read(mode='full')
        self.assertIsInstance(dataset, ArrowBasedDataset)
        self.assertEqual(dataset.mode, 'full')

        # 测试完整数据读取
        full_data = next(iter(dataset))
        self.assertEqual(len(full_data), 100)  # 应该一次性读取所有 100 行

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.reader.read(mode='invalid_mode')

    def test_file_not_found(self):
        self.mock_dataset.storage.exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            self.reader.read()

    def test_features(self):
        dataset = self.reader.read(mode='full')
        features = dataset.features
        self.assertIn('id', features)
        self.assertIn('value', features)
        self.assertEqual(str(features['id'].type), 'struct<id: int64>')
        self.assertEqual(str(features['value'].type), 'struct<value: string>')

    @patch('milvus_dataset.reader.pq.ParquetFile')
    def test_memory_usage(self, mock_parquet_file):
        # 创建一个模拟的 schema
        mock_schema = MagicMock()
        mock_schema.names = ['id', 'value']
        mock_field_id = MagicMock()
        mock_field_id.name = 'id'
        mock_field_id.type = pa.int64()
        mock_field_value = MagicMock()
        mock_field_value.name = 'value'
        mock_field_value.type = pa.string()
        mock_schema.__iter__.return_value = iter([mock_field_id, mock_field_value])

        # 模拟一个大文件，但只返回少量数据
        mock_pf = MagicMock()
        mock_pf.metadata.num_rows = 1000000
        mock_pf.schema_arrow = mock_schema
        mock_pf.read_row_group.return_value = pa.table({'id': [0], 'value': ['test']})
        mock_pf.iter_batches.return_value = [pa.table({'id': [0], 'value': ['test']})]
        mock_parquet_file.return_value = mock_pf

        dataset = self.reader.read(mode='stream')

        # 验证是否使用了流式读取（不会一次性加载所有数据）
        list(dataset)  # 触发数据读取
        mock_pf.iter_batches.assert_called()  # 确保调用了 read_row_group 而不是一次性读取全部


if __name__ == '__main__':
    unittest.main()
