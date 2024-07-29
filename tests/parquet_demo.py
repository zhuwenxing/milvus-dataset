import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


# 创建示例数据和Parquet文件
def create_sample_parquet(filename='example.parquet', num_rows=10000):
    # 创建示例数据
    data = {
        'id': np.arange(num_rows),
        'value': np.random.rand(num_rows),
        'category': np.random.choice(['A', 'B', 'C'], num_rows)
    }
    table = pa.Table.from_pydict(data)

    # 写入Parquet文件
    pq.write_table(table, filename)
    print(f"Created sample Parquet file: {filename}")


# 1. Stream模式
def read_stream(filename):
    print("\nStream Mode:")
    with pq.ParquetFile(filename) as pf:
        for batch in pf.iter_batches():
            process_batch(batch)


# 2. Batch模式
def read_batch(filename, batch_size=1000):
    print(f"\nBatch Mode (batch size: {batch_size}):")
    with pq.ParquetFile(filename) as pf:
        for batch in pf.iter_batches(batch_size=batch_size):
            process_batch(batch)


# 3. Full模式
def read_full(filename):
    print("\nFull Mode:")
    table = pq.read_table(filename)
    process_table(table)


# 示例处理函数
def process_batch(batch):
    print(f"Processing batch with {len(batch)} rows")
    # 这里可以添加更多的处理逻辑
    # 例如：print(batch.to_pydict())


def process_table(table):
    print(f"Processing table with {len(table)} rows")
    # 这里可以添加更多的处理逻辑
    # 例如：print(table.to_pydict())


# 主函数
def main():
    filename = 'example.parquet'

    # 创建示例Parquet文件
    create_sample_parquet(filename)

    # 演示三种读取模式
    read_stream(filename)
    read_batch(filename, batch_size=2000)
    read_full(filename)


if __name__ == "__main__":
    main()
