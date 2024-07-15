import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_neighbors(dataset, query_data, filter_condition=None, use_gpu=False):
    data = dataset.read(mode='full')
    if filter_condition:
        data = data.query(filter_condition)

    embeddings = data['embedding'].tolist()  # Assuming 'embedding' column exists

    if use_gpu:
        try:
            import cuml
            nn = cuml.neighbors.NearestNeighbors()
        except ImportError:
            print("cuML not available. Falling back to CPU.")
            nn = NearestNeighbors()
    else:
        nn = NearestNeighbors()

    nn.fit(embeddings)
    distances, indices = nn.kneighbors(query_data)

    return distances, indices
