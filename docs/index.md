# Milvus Dataset

A Python library for managing datasets in Milvus vector database.

## Features

- Easy dataset management for Milvus
- Efficient data loading and processing
- Support for various data formats
- Built-in neighbor computation

## Installation

```bash
pip install milvus-dataset
```

## Quick Start

```python
from milvus_dataset import LocalDataset

# Create a dataset
dataset = LocalDataset("my_dataset")

# Add data
dataset.add(vectors=vectors, metadatas=metadatas)

# Upload to Milvus
dataset.to_milvus("collection_name")
```

For more detailed information, check out our [Getting Started](getting-started.md) guide.
