from sklearn.feature_extraction.text import TfidfVectorizer

def generate_sparse_embeddings(dataset, text_column: str, model_name: str):
    if model_name != 'tfidf':
        raise ValueError("Only 'tfidf' is supported for sparse embeddings currently.")
    texts = dataset.read(mode='full')[text_column].tolist()
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts)
    return embeddings
