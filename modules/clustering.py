import numpy as np
from sklearn.cluster import KMeans

def run_kmeans(features, land_mask, H, W, n_clusters=4):
    """
    对特征数据在土地掩膜区域内执行 KMeans 聚类，并返回重映射后的聚类结果图像。

    参数:
        features (np.ndarray): 所有像素的特征向量，形状为 (H*W, n_features)。
        land_mask (np.ndarray): 布尔型土地掩膜，形状为 (H, W)。
        H, W (int): 图像的高和宽。
        n_clusters (int): 聚类数量，默认值为 4。

    返回:
        np.ndarray: 聚类后的标签图像，形状为 (H, W)，未分类区域为 -1。
    """
    land_features = features[land_mask.flatten()]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(land_features)

    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    remap_dict = {unique_labels[idx]: len(unique_labels) - i for i, idx in enumerate(sorted_indices)}
    sorted_labels = np.vectorize(remap_dict.get)(predicted_labels)

    kmeans_labels = np.full((H, W), -1)
    kmeans_labels[land_mask] = sorted_labels
    return kmeans_labels
