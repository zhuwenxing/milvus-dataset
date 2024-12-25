# Milvus Dataset

Milvus Dataset is a versatile Python library for efficient management and processing of large-scale datasets. While optimized for seamless integration with Milvus vector database, it also serves as a powerful standalone dataset management tool. The library provides a simple yet powerful interface for creating, writing, reading, and managing datasets, particularly excelling in handling large-scale vector data and general-purpose data management tasks.

## Key Features

1. **Flexible Storage Support**
   - Local storage support
   - Object storage support (S3/MinIO)
   - Easy migration between different storage types

2. **Rich Data Type Support**
   - Basic data types (INT64, VARCHAR, etc.)
   - Vector data types (FLOAT_VECTOR)
   - JSON fields
   - Sparse vectors
   - Binary vectors

3. **Dataset Management**
   - Training and test set split support
   - Dataset metadata management
   - Dataset statistics and analytics
   - Schema definition and validation

4. **Integration Capabilities**
   - Import to Milvus database
   - Upload to Hugging Face Hub
   - Seamless pandas DataFrame integration
   - Built-in nearest neighbor computation
   - Built-in mock data generation

## Installation

```bash
pip install milvus-dataset
```

## Quick Start Guide

### 1. Basic Configuration

```python
from milvus_dataset import ConfigManager, StorageType

# Initialize local storage
ConfigManager().init_storage(
    root_path="./data/my-dataset",
    storage_type=StorageType.LOCAL,
)

# Initialize S3 storage
ConfigManager().init_storage(
    root_path="s3://bucket/path",
    storage_type=StorageType.S3,
    options={
        "aws_access_key_id": "your_key",
        "aws_secret_access_key": "your_secret",
        "endpoint_url": "your_endpoint"  # Optional, for MinIO
    }
)
```

### 2. Creating a Dataset

```python
from pymilvus import CollectionSchema, DataType, FieldSchema
from milvus_dataset import load_dataset

# Define Schema
schema = CollectionSchema(
    fields=[
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1024)
    ],
    description="Text vector dataset"
)

# Load dataset
dataset = load_dataset("my-dataset", schema=schema)
```

### 3. Writing Data

```python
import pandas as pd
import numpy as np

# Prepare data
df = pd.DataFrame({
    "id": range(1000),
    "text": ["text_" + str(i) for i in range(1000)],
    "embedding": [np.random.rand(1024) for _ in range(1000)]
})

# Write to training set
with dataset["train"].get_writer(mode="append") as writer:
    writer.write(df)
```

### 4. Dataset Operations

```python
# View dataset information
print(dataset.summary())

# Compute neighbors
dataset.compute_neighbors(
    vector_field_name="embedding",
    pk_field_name="id",
    top_k=100
)

# import to Milvus
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

# Upload to Hugging Face
dataset.to_hf(repo_name="username/dataset-name")
```

## Advanced Usage

### Performance Optimization

1. **File Size Configuration**
   ```python
   with dataset["train"].get_writer(
       mode="append",
       target_file_size_mb=512,  # Adjust file size
       num_buffers=15,           # Adjust buffer number
       queue_size=30             # Adjust queue size
   ) as writer:
       writer.write(df)
   ```

2. **Batch Processing**
   ```python
   # Read in batches
   for batch in dataset["train"].read(mode="batch", batch_size=1000):
       process_batch(batch)
   ```

### Storage Migration

```python
# Move data from local to S3
dataset.to_storage(StorageConfig(
    storage_type=StorageType.S3,
    root_path="s3://bucket/path",
    options={...}
))
```

## Common Issues and Solutions

1. **Storage Type Selection**
   - Use local storage for development and testing
   - Use object storage for production environments

2. **Handling Large-Scale Data**
   - Use batch writing
   - Set appropriate buffer size and queue size
   - Consider parallel processing

3. **Ensuring Data Quality**
   - Define comprehensive schema
   - Enable schema validation
   - Regularly check dataset statistics

4. **Performance Optimization Tips**
   - Set reasonable file size (target_file_size_mb)
   - Adjust buffer parameters (num_buffers, queue_size)
   - Process data in batches instead of one by one

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.
