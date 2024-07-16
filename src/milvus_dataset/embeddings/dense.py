from fastembed import TextEmbedding
from typing import List

def generate_dense_embeddings(dataset, text_column: str, model_name: str = "BAAI/bge-small-en-v1.5"):
    embedding_model = TextEmbedding(model_name)
    texts: List[str] = dataset.read(mode='full')[text_column].tolist()
    embeddings = list(embedding_model.embed(texts))
    return embeddings
