# Milvus Dataset

Milvus Dataset 是一个用于高效管理和处理大规模数据集的多功能 Python 库。虽然针对 Milvus 向量数据库进行了优化，但它同时也是一个强大的独立数据集管理工具。该库提供了简单而强大的接口来创建、写入、读取和管理数据集，不仅在处理大规模向量数据方面表现出色，在通用数据管理任务中也同样高效实用。

## 主要特性

1. **灵活的存储支持**
   - 支持本地存储
   - 支持对象存储(S3/MinIO)
   - 支持在不同存储之间迁移数据集

2. **丰富的数据类型支持**
   - 支持基础数据类型(INT64, VARCHAR等)
   - 支持向量数据类型(FLOAT_VECTOR)
   - 支持JSON字段
   - 支持稀疏向量
   - 支持二进制向量

3. **数据集管理功能**
   - 支持训练集和测试集的分割
   - 支持数据集元信息管理
   - 支持数据集统计信息查看
   - 支持数据集模式(Schema)定义和验证

4. **集成能力**
   - 支持导入到Milvus数据库
   - 支持上传到Hugging Face Hub
   - 支持与pandas DataFrame无缝集成
   - 内置近邻计算功能
   - 内置模拟数据生成

## 安装

```bash
pip install milvus-dataset
```

## 快速入门指南

### 1. 基础配置

```python
from milvus_dataset import ConfigManager, StorageType

# 初始化本地存储
ConfigManager().init_storage(
    root_path="./data/my-dataset",
    storage_type=StorageType.LOCAL,
)

# 初始化S3存储
ConfigManager().init_storage(
    root_path="s3://bucket/path",
    storage_type=StorageType.S3,
    options={
        "aws_access_key_id": "your_key",
        "aws_secret_access_key": "your_secret",
        "endpoint_url": "your_endpoint"  # 可选，用于MinIO
    }
)
```

### 2. 创建数据集

```python
from pymilvus import CollectionSchema, DataType, FieldSchema
from milvus_dataset import load_dataset

# 定义Schema
schema = CollectionSchema(
    fields=[
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1024)
    ],
    description="文本向量数据集"
)

# 加载数据集
dataset = load_dataset("my-dataset", schema=schema)
```

### 3. 写入数据

```python
import pandas as pd
import numpy as np

# 准备数据
df = pd.DataFrame({
    "id": range(1000),
    "text": ["text_" + str(i) for i in range(1000)],
    "embedding": [np.random.rand(1024) for _ in range(1000)]
})

# 写入训练集
with dataset["train"].get_writer(mode="append") as writer:
    writer.write(df)
```

### 4. 数据集操作

```python
# 查看数据集信息
print(dataset.summary())

# 计算近邻
dataset.compute_neighbors(
    vector_field_name="embedding",
    pk_field_name="id",
    top_k=100
)

# 导出到Milvus
dataset.to_milvus(
    milvus_config={
        "host": "localhost",
        "port": 19530
    },
    milvus_storage=StorageConfig(
        root_path="s3://bucket/path",
        storage_type=StorageType.S3,
        options={
            "aws_access_key_id": "your_key",
            "aws_secret_access_key": "your_secret",
            "endpoint_url": "your_endpoint"  # Optional, for MinIO
        }
    )
)

# 上传到Hugging Face
dataset.to_hf(repo_name="username/dataset-name")
```

## 高级用法

### 性能优化

1. **文件大小配置**
   ```python
   with dataset["train"].get_writer(
       mode="append",
       target_file_size_mb=512,  # 调整文件大小
       num_buffers=15,           # 调整缓冲区数量
       queue_size=30             # 调整队列大小
   ) as writer:
       writer.write(df)
   ```

2. **批量处理**
   ```python
   # 批量读取
   for batch in dataset["train"].read(mode="batch", batch_size=1000):
       process_batch(batch)
   ```

### 存储迁移

```python
# 将数据从本地迁移到S3
dataset.to_storage(StorageConfig(
    storage_type=StorageType.S3,
    root_path="s3://bucket/path",
    options={...}
))
```

## 常见问题与解决方案

1. **如何选择存储类型？**
   - 开发测试时使用本地存储
   - 生产环境建议使用对象存储

2. **如何处理大规模数据？**
   - 使用批量写入
   - 设置合适的buffer size和queue size
   - 考虑使用并行处理

3. **如何确保数据质量？**
   - 定义完善的schema
   - 启用schema验证
   - 定期检查数据集统计信息

4. **性能优化建议**
   - 合理设置文件大小(target_file_size_mb)
   - 调整缓冲区参数(num_buffers, queue_size)
   - 批量处理数据而不是逐条处理

## 贡献

我们欢迎各种形式的贡献！如果您发现了bug或有功能建议，请创建issue。如果您想贡献代码，请提交pull request。
