---
sidebar_position: 1
---

# Quickstart Guide

This guide will help you get started with Milvus Dataset quickly.

## Installation

```bash
pip install milvus-dataset
```

## Basic Usage

Here's a simple example of how to create a dataset and load it into Milvus:

```python
from milvus_dataset import Dataset

# Create a dataset
dataset = Dataset("example_collection")

# Add some vector data
import numpy as np
vectors = np.random.rand(1000, 128)  # 1000 vectors of dimension 128
metadata = [{"id": i} for i in range(1000)]

# Add data to dataset
dataset.add_data(vectors, metadata)

# Load into Milvus
dataset.to_milvus(
    connection_args={
        "host": "localhost",
        "port": 19530
    }
)
```

## Next Steps

Check out our other tutorials to learn more about:
- [Data Transformation](./transform-data.md)
- [Advanced Features](./advanced-features.md)
- [Best Practices](./best-practices.md)
