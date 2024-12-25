# Neighbors Computation Implementation Guide

This document explains the implementation details of the neighbors computation functionality in Milvus Dataset.

## Overview

The neighbors computation module provides efficient nearest neighbor search capabilities with the following key features:

### 1. Multiple Acceleration Methods

#### GPU Acceleration
```python
if GPU_AVAILABLE:
    # Use CUDA for fast computation
    distances, indices = knn(
        train_emb_gpu, test_emb_gpu, 
        k=self.top_k, 
        metric=self.metric_type
    )
```

#### Numba Acceleration
```python
@numba.jit(nopython=True, parallel=True)
def compute_distances(vectors_a, vectors_b):
    # Accelerated distance computation
    return distances
```

### 2. MapReduce for Large Datasets

The module handles large datasets by splitting computation into multiple epochs:

```python
def compute_ground_truth(self):
    # Split data into epochs
    num_epochs = self._calculate_num_epochs()
    
    for epoch in range(num_epochs):
        # Map: Compute partial results
        partial_results = self.compute_neighbors(
            test_data[start_idx:end_idx],
            train_data[start_idx:end_idx]
        )
        
        # Save partial results
        save_partial_results(partial_results)
    
    # Reduce: Merge all partial results
    final_results = self.merge_neighbors(partial_files)
```

### 3. Distance Metrics Support

```python
neighbors = dataset.compute_neighbors(
    vector_field_name="embedding",
    metric_type="cosine",  # or "euclidean", "dot_product", etc.
    top_k=1000
)
```

## Key Features

1. **Acceleration Options**
   - GPU acceleration with CUDA
   - CPU acceleration with Numba
   - Automatic fallback mechanisms

2. **Large Dataset Support**
   - MapReduce-style processing
   - Memory-efficient computation
   - Partial result handling

3. **Flexible Distance Metrics**
   - Cosine similarity
   - Euclidean distance
   - Dot product
   - Custom metrics support
   - Support scalar filters
