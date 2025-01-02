# Getting Started

This guide will help you get started with Milvus Dataset.

## Prerequisites

- Python 3.8 or later
- A running Milvus instance

## Installation

Install using pip:

```bash
pip install milvus-dataset
```

## Basic Usage

### Creating a Dataset

```python
from milvus_dataset import LocalDataset

# Create a new dataset
dataset = LocalDataset("my_dataset")
```

### Adding Data

```python
import numpy as np

# Generate some example data
vectors = np.random.rand(1000, 128)
metadatas = [{"text": f"document_{i}"} for i in range(1000)]

# Add to dataset
dataset.add(vectors=vectors, metadatas=metadatas)
```

### Uploading to Milvus

```python
# Upload to Milvus
dataset.to_milvus(
    collection_name="my_collection",
    dim=128,
    metric_type="L2"
)
```

## Next Steps

- Check out the [API Reference](api/core.md) for detailed documentation
- See [Examples](examples.md) for more usage examples
