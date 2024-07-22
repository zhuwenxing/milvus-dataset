# Milvus Dataset

Milvus Dataset is a Python library designed for efficient management and processing of large-scale datasets, specifically tailored for integration with Milvus vector database. It provides a simple yet powerful interface for creating, writing, reading, and managing datasets, particularly suited for handling large-scale vector data.

## Key Features

- **Intelligent File Splitting**: Automatically splits large datasets into appropriately sized files, optimizing storage and query efficiency.
- **Flexible Data Format Support**: Supports various data formats including pandas DataFrame, PyArrow Table, dictionaries, and lists of dictionaries.
- **Efficient Data Writing**: Utilizes Dask for parallelized data writing, significantly enhancing large-scale data processing speed.
- **Dynamic File Size Adjustment**: Automatically adjusts file sizes to ensure optimal storage and query performance.
- **Seamless Milvus Integration**: Designed specifically for Milvus vector database, supporting efficient vector data management and querying.
- **Multiple Reading Modes**: Supports streaming, batch, and full data reading, adapting to different use cases.
- **Data Validation**: Offers optional schema validation for training datasets, ensuring data quality.

## Installation

Install Milvus Dataset using pip:

```bash
pip install milvus-dataset
```

## Quick Start

Here's a simple usage example:

```python
from milvus_dataset import Dataset, configure_logger

# Configure logging level
configure_logger(level="INFO")

# Initialize the dataset
dataset = Dataset("my_dataset", root_path="/path/to/data")

# Write data
data = {...}  # Your data, can be a DataFrame, dictionary, etc.
dataset.write(data, mode='append')

# Read data
train_data = dataset.read(split='train')
```

## Detailed Usage

### Writing Data

```python
# Use DatasetWriter for more granular control
from milvus_dataset import DatasetWriter

writer = DatasetWriter(dataset, target_file_size_mb=5)
writer.write(data, mode='append')
```

### Reading Data

```python
# Full read
full_data = dataset.read(mode='full')

# Stream read
for batch in dataset.read(mode='stream'):
    process_batch(batch)

# Batch read
for batch in dataset.read(mode='batch', batch_size=1000):
    process_batch(batch)
```

### Schema Validation

```python
# Set schema for training data
dataset.set_schema({
    "id": (int, ...),
    "vector": ([float], 128),  # 128-dimensional vector
    "label": (str, ...)
})
```

## Configuration

Milvus Dataset can be configured through environment variables or a configuration file. Key configuration items include:

- `MILVUS_DATASET_ROOT`: Root directory for datasets
- `MILVUS_DATASET_LOG_LEVEL`: Logging level

## Contributing

We welcome contributions of all forms! If you find a bug or have a feature suggestion, please create an issue. If you'd like to contribute code, please submit a pull request.

## License

Milvus Dataset is licensed under the [Apache 2.0 License](LICENSE).

## Contact Us

If you have any questions or suggestions, please contact us through [GitHub Issues](https://github.com/your-repo/milvus-dataset/issues).

