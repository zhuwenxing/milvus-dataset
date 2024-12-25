# 近邻计算实现指南

本文档说明了 Milvus Dataset 中近邻计算功能的实现细节。

## 概述

近邻计算模块提供高效的最近邻搜索功能，具有以下关键特性：

### 1. 多种加速方法

#### GPU 加速
```python
if GPU_AVAILABLE:
    # 使用 CUDA 进行快速计算
    distances, indices = knn(
        train_emb_gpu, test_emb_gpu, 
        k=self.top_k, 
        metric=self.metric_type
    )
```

#### Numba 加速
```python
@numba.jit(nopython=True, parallel=True)
def compute_distances(vectors_a, vectors_b):
    # 加速距离计算
    return distances
```

### 2. 大数据集的 MapReduce 处理

模块通过将计算分割为多个轮次来处理大型数据集：

```python
def compute_ground_truth(self):
    # 将数据分割为多个轮次
    num_epochs = self._calculate_num_epochs()
    
    for epoch in range(num_epochs):
        # Map：计算部分结果
        partial_results = self.compute_neighbors(
            test_data[start_idx:end_idx],
            train_data[start_idx:end_idx]
        )
        
        # 保存部分结果
        save_partial_results(partial_results)
    
    # Reduce：合并所有部分结果
    final_results = self.merge_neighbors(partial_files)
```

### 3. 距离度量支持

```python
neighbors = dataset.compute_neighbors(
    vector_field_name="embedding",
    metric_type="cosine",  # 或 "euclidean"、"dot_product" 等
    top_k=1000
)
```

## 主要特性

1. **加速选项**
   - GPU 加速（CUDA）
   - CPU 加速（Numba）
   - 自动回退机制

2. **大数据集支持**
   - MapReduce 式处理
   - 内存高效计算
   - 部分结果处理

3. **灵活的距离度量**
   - 余弦相似度
   - 欧氏距离
   - 点积
   - 支持自定义度量
   - 支持标量过滤
