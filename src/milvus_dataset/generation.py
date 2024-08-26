import numpy as np
from typing import List
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import time

# 设置参数
n_samples_original = 1000  # 原始数据样本数
n_features = 768  # 特征维度
n_components_pca = 50  # PCA降维后的维度
n_components_gmm = 5  # GMM的组件数
n_samples_generate = 100000  # 要生成的样本数

# 生成原始数据
np.random.seed(0)
X = np.random.randn(n_samples_original, n_features)

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维
pca = PCA(n_components=n_components_pca)
X_pca = pca.fit_transform(X_scaled)

# 训练GMM
gmm = GaussianMixture(n_components=n_components_gmm, random_state=0)
gmm.fit(X_pca)

# 生成新样本并计时
start_time = time.time()

# 分批生成数据以避免内存问题
batch_size = 100000
generated_samples = []

for i in range(0, n_samples_generate, batch_size):
    batch_samples_pca = gmm.sample(batch_size)[0]
    batch_samples = pca.inverse_transform(batch_samples_pca)
    batch_samples = scaler.inverse_transform(batch_samples)
    generated_samples.append(batch_samples)

# 合并所有批次的数据
new_samples = np.vstack(generated_samples)

end_time = time.time()

# 计算并打印结果
total_time = end_time - start_time
print(f"生成 {n_samples_generate} 条 {n_features} 维向量数据耗时: {total_time:.2f} 秒")
print(f"平均每秒生成 {n_samples_generate / total_time:.2f} 条数据")
print(f"生成的数据形状: {new_samples.shape}")


def generate_vector_list(dim: int, num_entities: int, cluster_num: int) -> List[List[float]]:
    np.random.seed(42)
    
    # Create all points at once
    points_per_cluster = num_entities // cluster_num
    remainder = num_entities % cluster_num
    
    # Create cluster centers
    centers = np.random.randn(cluster_num, dim) * 10
    
    # Generate all points in one go
    vectors = np.vstack([
        np.random.randn(points_per_cluster + (1 if i < remainder else 0), dim) * 2 + centers[i]
        for i in range(cluster_num)
    ])
    
    # Shuffle the vectors
    np.random.shuffle(vectors)
    
    # # Standardize the vectors
    # scaler = StandardScaler()
    # vectors_scaled = scaler.fit_transform(vectors)
    
    return vectors.tolist()


if __name__ == "__main__":
    generate_vector_list(768, 100000, 5)
    # 可视化这些向量
    # 可视化生成的向量

    def visualize_vectors(vectors: List[List[float]], n_samples: int = 1000):
        import matplotlib.pyplot as plt
        import random
        from sklearn.decomposition import PCA

        # 如果向量数量超过n_samples，随机选择n_samples个向量
        if len(vectors) > n_samples:
            vectors = random.sample(vectors, n_samples)

        # 将向量列表转换为numpy数组
        vectors_array = np.array(vectors)

        # 使用PCA将向量降至2维
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors_array)

        # 绘制散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
        plt.title("Vector Visualization (PCA)")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.show()

    # 生成向量并可视化
    dim = 768
    num_entities = 100000
    cluster_num = 5
    t0 = time.time()
    vectors = generate_vector_list(dim, num_entities, cluster_num)
    t1 = time.time()
    print(f"生成 {num_entities} 条 {dim} 维向量数据耗时: {t1 - t0:.2f} 秒")
    # visualize_vectors(vectors)
