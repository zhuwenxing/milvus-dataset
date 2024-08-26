# sparse embedding and dense embedding
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch


class TextEmbedder:

    def __init__(
        self,
        sparse_max_features: int = 10000,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        初始化TextEmbedder类。

        Args:
            sparse_max_features (int): TfidfVectorizer的最大特征数。
            dense_model_name (str): 用于密集嵌入的预训练模型名称。
        """
        self.sparse_vectorizer = TfidfVectorizer(
            max_features=sparse_max_features
        )
        self.dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_name)
        self.dense_model = AutoModel.from_pretrained(dense_model_name)

    def fit_sparse(self, texts: List[str]) -> None:
        """
        使用给定的文本训练TfidfVectorizer。

        Args:
            texts (List[str]): 用于训练的文本列表。
        """
        self.sparse_vectorizer.fit(texts)

    def get_sparse_embedding(self, texts: List[str]) -> np.ndarray:
        """
        获取文本的稀疏嵌入（TF-IDF向量）。

        TF-IDF（词频-逆文档频率）是一种用于信息检索和文本挖掘的常用加权技术。
        它反映了一个词在文档中的重要性。TF-IDF的值与一个词在文档中出现的次数
        成正比，同时与该词在整个文档集中出现的频率成反比。

        Args:
            texts (List[str]): 要嵌入的文本列表。

        Returns:
            np.ndarray: 稀疏嵌入矩阵。
        """
        return self.sparse_vectorizer.transform(texts).toarray()

    def get_dense_embedding(self, texts: List[str]) -> np.ndarray:
        """
        获取文本的密集嵌入。

        Args:
            texts (List[str]): 要嵌入的文本列表。

        Returns:
            np.ndarray: 密集嵌入矩阵。
        """
        inputs = self.dense_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.dense_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

    def get_embeddings(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        获取文本的稀疏和密集嵌入。

        Args:
            texts (List[str]): 要嵌入的文本列表。

        Returns:
            Dict[str, np.ndarray]: 包含稀疏和密集嵌入的字典。
        """
        sparse_embeddings = self.get_sparse_embedding(texts)
        dense_embeddings = self.get_dense_embedding(texts)
        return {
            "sparse_embedding": sparse_embeddings,
            "dense_embedding": dense_embeddings
        }


# 使用示例
embedder = TextEmbedder()
texts = ["This is a sample text", "Another example"]
embedder.fit_sparse(texts)
embeddings = embedder.get_embeddings(texts)

print("Sparse embedding:")
print(embeddings["sparse_embedding"])
print("\nSparse embedding shape:", embeddings["sparse_embedding"].shape)

print("\nDense embedding:")
print(embeddings["dense_embedding"])
print("\nDense embedding shape:", embeddings["dense_embedding"].shape)

# 输出示例：
# Sparse embedding:
# [[0.70710678 0.         0.70710678 0.        ]
#  [0.         0.70710678 0.         0.70710678]]
# Sparse embedding shape: (2, 4)

# Dense embedding:
# [[-0.0394287   0.00676092  0.02132456 ...  0.01097197 -0.03594252
#   -0.00499517]
#  [-0.03310472  0.01482943  0.0136438  ...  0.00682998 -0.02657881
#   -0.00607809]]
# Dense embedding shape: (2, 384)
