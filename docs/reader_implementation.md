# Dataset Reader Implementation Guide

This document explains the implementation details of the dataset reader in Milvus Dataset.

## Overview

The dataset reader supports two reading modes to accommodate different data processing needs:

### 1. Full Mode
```python
# Load entire dataset into memory at once
df = dataset["train"].read(mode="full")
```
- Loads the complete dataset into memory
- Suitable for smaller datasets
- Simple to use when memory is sufficient

### 2. Stream Mode
```python
# Stream data in batches
for batch in dataset["train"].read(mode="stream", batch_size=1000):
    process_batch(batch)
```
- Processes data in batches using generators
- Memory efficient for large datasets
- Allows processing of datasets larger than available RAM

## Usage Example

```python
# Choose mode based on your needs
if dataset_size < available_memory:
    # Full mode for small datasets
    df = dataset["train"].read(mode="full")
    process_data(df)
else:
    # Stream mode for large datasets
    for batch in dataset["train"].read(mode="stream", batch_size=1000):
        process_batch(batch)
