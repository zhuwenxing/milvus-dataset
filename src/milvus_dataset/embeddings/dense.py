from sentence_transformers import SentenceTransformer

def generate_dense_embeddings(dataset, text_column: str, model_name: str):
    model = SentenceTransformer(model_name)
    texts = dataset.read(mode='full')[text_column].tolist()
    embeddings = model.encode(texts)
    return embeddings
