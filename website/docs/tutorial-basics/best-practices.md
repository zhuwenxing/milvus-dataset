---
sidebar_position: 4
---

# Best Practices

Learn the best practices for using Milvus Dataset effectively in your applications.

## Data Preparation

### Clean Your Data
```python
from milvus_dataset import Dataset
import numpy as np

def clean_data(vectors, metadata):
    # Remove null values
    valid_indices = ~np.isnan(vectors).any(axis=1)
    clean_vectors = vectors[valid_indices]
    clean_metadata = [m for i, m in enumerate(metadata) if valid_indices[i]]
    return clean_vectors, clean_metadata

# Use cleaned data
dataset = Dataset("clean_collection")
clean_vectors, clean_metadata = clean_data(vectors, metadata)
dataset.add_data(clean_vectors, clean_metadata)
```

### Normalize Vectors
```python
from sklearn.preprocessing import normalize

# Normalize vectors before adding to dataset
normalized_vectors = normalize(vectors)
dataset.add_data(normalized_vectors, metadata)
```

## Performance Optimization

### Batch Processing
```python
def process_in_batches(dataset, vectors, metadata, batch_size=1000):
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]
        dataset.add_data(batch_vectors, batch_metadata)
```

### Resource Management
```python
# Configure resource usage
dataset = Dataset(
    "optimized_collection",
    batch_size=1000,
    num_workers=4,
    cache_size="2GB"
)

# Use context manager for proper cleanup
with dataset:
    dataset.add_data(vectors, metadata)
    dataset.to_milvus()
```

## Error Handling

### Implement Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_add_data(dataset, vectors, metadata):
    try:
        dataset.add_data(vectors, metadata)
    except Exception as e:
        logger.error(f"Error adding data: {e}")
        raise
```

### Validation
```python
def validate_vectors(vectors, expected_dim=128):
    if not isinstance(vectors, np.ndarray):
        raise ValueError("Vectors must be numpy array")
    if len(vectors.shape) != 2:
        raise ValueError("Vectors must be 2-dimensional")
    if vectors.shape[1] != expected_dim:
        raise ValueError(f"Vector dimension must be {expected_dim}")
```

## Monitoring and Logging

### Set Up Logging
```python
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("milvus_dataset")

logger = setup_logging()
```

### Track Operations
```python
from contextlib import contextmanager
import time

@contextmanager
def track_operation(operation_name):
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"{operation_name} completed in {duration:.2f} seconds")

# Use tracking
with track_operation("data_loading"):
    dataset.add_data(vectors, metadata)
```

## Security

### Handle Sensitive Data
```python
import hashlib

def hash_sensitive_metadata(metadata):
    """Hash sensitive fields in metadata"""
    for item in metadata:
        if 'sensitive_field' in item:
            item['sensitive_field'] = hashlib.sha256(
                item['sensitive_field'].encode()
            ).hexdigest()
    return metadata

# Use hashed metadata
hashed_metadata = hash_sensitive_metadata(metadata)
dataset.add_data(vectors, hashed_metadata)
```

## Backup and Recovery

### Implement Backup Strategy
```python
def backup_dataset(dataset, backup_path):
    """Save dataset state to disk"""
    dataset.save_state(backup_path)
    logger.info(f"Dataset backed up to {backup_path}")

def restore_dataset(backup_path):
    """Restore dataset from backup"""
    dataset = Dataset.load_state(backup_path)
    logger.info(f"Dataset restored from {backup_path}")
    return dataset
```
