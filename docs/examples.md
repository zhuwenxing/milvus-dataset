# Examples

## Basic Dataset Operations

```python
from milvus_dataset import LocalDataset
import numpy as np

# Create dataset
dataset = LocalDataset("my_dataset")

# Add data
vectors = np.random.rand(1000, 128)
metadatas = [{"text": f"doc_{i}"} for i in range(1000)]
dataset.add(vectors=vectors, metadatas=metadatas)

# Save dataset
dataset.save()

# Load dataset
loaded_dataset = LocalDataset.load("my_dataset")

# Upload to Milvus
loaded_dataset.to_milvus("my_collection")
```

## Computing Neighbors

```python
from milvus_dataset.neighbors import compute_neighbors

# Compute neighbors
neighbors = compute_neighbors(
    queries=query_vectors,
    index_vectors=index_vectors,
    k=10
)

# Access results
for i, (indices, distances) in enumerate(zip(neighbors.indices, neighbors.distances)):
    print(f"Query {i} nearest neighbors:")
    for j, (idx, dist) in enumerate(zip(indices, distances)):
        print(f"  {j+1}. Index: {idx}, Distance: {dist}")
```
