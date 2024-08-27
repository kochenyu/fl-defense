from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
X = np.random.rand(100, 2)

# 创建 KMeans 模型
kmeans = KMeans(n_clusters=3, random_state=0)

# 训练模型
kmeans.fit(X)

# 获取簇标签
labels = kmeans.labels_

# 获取簇中心
centroids = kmeans.cluster_centers_

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
