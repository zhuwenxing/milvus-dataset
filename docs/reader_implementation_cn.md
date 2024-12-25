# 数据集读取器实现指南

本文档说明了 Milvus Dataset 中数据集读取器的实现细节。

## 概述

数据集读取器支持两种读取模式以适应不同的数据处理需求：

### 1. 完整模式
```python
# 一次性将整个数据集加载到内存
df = dataset["train"].read(mode="full")
```
- 将完整数据集加载到内存中
- 适用于较小的数据集
- 内存充足时使用简单

### 2. 流式模式
```python
# 批量流式读取数据
for batch in dataset["train"].read(mode="stream", batch_size=1000):
    process_batch(batch)
```
- 使用生成器批量处理数据
- 大数据集的内存高效处理
- 允许处理超过可用内存的数据集

## 使用示例

```python
# 根据需求选择模式
if dataset_size < available_memory:
    # 小数据集使用完整模式
    df = dataset["train"].read(mode="full")
    process_data(df)
else:
    # 大数据集使用流式模式
    for batch in dataset["train"].read(mode="stream", batch_size=1000):
        process_batch(batch)
```
