---
sidebar_position: 2
---

# Data Transformation

Learn how to transform different types of data into vector embeddings using Milvus Dataset.

## Text Data

Transform text data into embeddings using pre-trained models:

```python
from milvus_dataset import Dataset

# Create dataset
dataset = Dataset("text_collection")

# Add text data
texts = [
    "This is a sample text",
    "Another example sentence",
    "More text data for embedding"
]

# Transform text to embeddings
dataset.transform_text(
    texts,
    model="sentence-transformers/all-MiniLM-L6-v2"
)
```

## Image Data

Transform image data into embeddings:

```python
# Create dataset for images
image_dataset = Dataset("image_collection")

# Add image paths
image_paths = [
    "path/to/image1.jpg",
    "path/to/image2.jpg"
]

# Transform images to embeddings
image_dataset.transform_images(
    image_paths,
    model="clip-vit-base-patch32"
)
```

## Custom Transformations

You can also define custom transformation functions:

```python
def custom_transform(data):
    # Your custom transformation logic here
    return transformed_vectors

dataset = Dataset("custom_collection")
dataset.transform(data, transform_fn=custom_transform)
```

## Best Practices

1. **Batch Processing**: For large datasets, process data in batches
2. **Model Selection**: Choose appropriate models for your use case
3. **Data Preprocessing**: Clean and normalize data before transformation
4. **Error Handling**: Implement proper error handling for failed transformations
