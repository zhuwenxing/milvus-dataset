# Milvus Dataset

Milvus Dataset 是一个用于高效管理和处理大规模数据集的 Python 库，专为与 Milvus 向量数据库集成而设计。它提供了简单而强大的接口来创建、写入、读取和管理数据集，特别适合处理大规模的向量数据。

## 主要特性

- **智能文件分割**: 自动将大型数据集分割成适当大小的文件，优化存储和查询效率。
- **灵活的数据格式支持**: 支持 pandas DataFrame、PyArrow Table、字典和字典列表等多种数据格式。
- **高效的数据写入**: 利用 Dask 实现并行化数据写入，显著提升大规模数据处理速度。
- **动态文件大小调整**: 自动调整文件大小，确保最佳的存储和查询性能。
- **与 Milvus 无缝集成**: 专为 Milvus 向量数据库设计，支持高效的向量数据管理和查询。
- **多种读取模式**: 支持流式、批量和全量数据读取，适应不同的使用场景。
- **数据验证**: 对训练数据集提供可选的 schema 验证，确保数据质量。

## 安装

使用 pip 安装 Milvus Dataset：

```bash
pip install milvus-dataset
```

## 快速开始

以下是一个简单的使用示例：

```python
from milvus_dataset import Dataset, configure_logger

# 配置日志级别
configure_logger(level="INFO")

# 初始化数据集
dataset = Dataset("my_dataset", root_path="/path/to/data")

# 写入数据
data = {...}  # 您的数据，可以是 DataFrame、字典等
dataset.write(data, mode='append')

# 读取数据
train_data = dataset.read(split='train')
```

## 详细使用

### 数据写入

```python
# 使用 DatasetWriter 进行更细粒度的控制
from milvus_dataset import DatasetWriter

writer = DatasetWriter(dataset, target_file_size_mb=5)
writer.write(data, mode='append')
```

### 数据读取

```python
# 全量读取
full_data = dataset.read(mode='full')

# 流式读取
for batch in dataset.read(mode='stream'):
    process_batch(batch)

# 批量读取
for batch in dataset.read(mode='batch', batch_size=1000):
    process_batch(batch)
```

### Schema 验证

```python
# 为训练数据设置 schema
dataset.set_schema({
    "id": (int, ...),
    "vector": ([float], 128),  # 128维向量
    "label": (str, ...)
})
```

## 配置

Milvus Dataset 可以通过环境变量或配置文件进行配置。主要配置项包括：

- `MILVUS_DATASET_ROOT`: 数据集根目录
- `MILVUS_DATASET_LOG_LEVEL`: 日志级别

## 贡献

我们欢迎任何形式的贡献！如果您发现了 bug 或有新的功能建议，请创建一个 issue。如果您想贡献代码，请提交 pull request。

## 许可证

Milvus Dataset 采用 [Apache 2.0 许可证](LICENSE)。

## 联系我们

如果您有任何问题或建议，请通过 [GitHub Issues](https://github.com/your-repo/milvus-dataset/issues) 与我们联系。

