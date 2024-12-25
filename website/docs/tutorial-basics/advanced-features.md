---
sidebar_position: 3
---

# Advanced Features

Explore advanced features and capabilities of Milvus Dataset.

## Asynchronous Operations

Use async operations for better performance:

```python
from milvus_dataset import Dataset
import asyncio

async def process_large_dataset():
    dataset = Dataset("async_collection")
    
    # Async data addition
    await dataset.add_data_async(vectors, metadata)
    
    # Async transformation
    await dataset.transform_text_async(texts)
    
    # Async Milvus loading
    await dataset.to_milvus_async()

# Run async operations
asyncio.run(process_large_dataset())
```

## Data Validation

Implement custom validation rules:

```python
from milvus_dataset import Dataset, ValidationRule

class DimensionCheck(ValidationRule):
    def validate(self, vectors):
        return all(len(v) == 128 for v in vectors)

dataset = Dataset("validated_collection")
dataset.add_validation_rule(DimensionCheck())
```

## Data Partitioning

Organize data into partitions:

```python
dataset = Dataset("partitioned_collection")

# Create partitions
dataset.create_partition("2023_data")
dataset.create_partition("2024_data")

# Add data to specific partitions
dataset.add_data_to_partition("2023_data", vectors_2023, metadata_2023)
dataset.add_data_to_partition("2024_data", vectors_2024, metadata_2024)
```

## Performance Optimization

Configure performance settings:

```python
dataset = Dataset(
    "optimized_collection",
    batch_size=1000,
    num_workers=4,
    cache_size="2GB"
)

# Enable compression
dataset.enable_compression(algorithm="zstd", level=3)

# Configure index building
dataset.build_index(
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 1024}
)
```

## Error Handling and Logging

Implement comprehensive error handling:

```python
import logging
from milvus_dataset import Dataset, DatasetError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("milvus_dataset")

try:
    dataset = Dataset("error_handled_collection")
    dataset.add_data(vectors, metadata)
except DatasetError as e:
    logger.error(f"Dataset operation failed: {e}")
    # Implement recovery logic
```
