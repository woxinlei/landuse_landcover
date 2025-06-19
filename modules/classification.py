from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from osgeo import ogr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
def train_classifier(X_train, y_train, method="random_forest"):
    if method == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif method == "svm":
        model = SVC(kernel="rbf", probability=True)
    else:
        raise ValueError("method 仅支持 'random_forest' 或 'svm'")
    model.fit(X_train, y_train)
    return model

import numpy as np
from .postprocessing import smooth_mask, remove_small_objects
def generate_labels(features_flat, features_scaled, target_mask, exclude_mask, road_mask, zero_mask, H, W, label_name=""):
    print(f"🧠 训练 {label_name} 分类器...")
    labels = np.full((H, W), -1)
    labels[target_mask] = 1
    labels[exclude_mask | road_mask | zero_mask] = 0

    mask_valid = labels >= 0
    X_train = features_scaled[mask_valid.flatten()]
    y_train = labels[mask_valid]

    model = train_classifier(X_train, y_train)
    y_pred_full = model.predict(features_scaled)
    pred_mask = y_pred_full.reshape(H, W) == 1

    pred_mask = smooth_mask(pred_mask, size=5)
    pred_mask = remove_small_objects(pred_mask, min_size=30)

    print(f"✅ {label_name} 分类完成")
    return pred_mask
import numpy as np

def generate_landcover_mask(city_mask, water_mask, road_mask, land_mask, land_labels):
    """
    根据城市、水体、道路掩膜和土地聚类结果生成最终土地掩膜。
    """
    H, W = land_labels.shape
    final_labels = np.full((H, W), -1)
    
    final_labels[city_mask] = 99
    final_labels[water_mask] = 100
    final_labels[road_mask] = 101
    final_labels[land_mask] = land_labels[land_mask]
    
    return final_labels
def supervised_classification(X, dataset, shapefile_path, H, W):
    # 读取 shapefile 获取 label 样本
    vector = ogr.Open(shapefile_path)
    layer = vector.GetLayer()

    # 创建掩膜图层，用于获取标签
    mask = np.zeros((H, W), dtype=np.uint8)

    from osgeo import gdal
    mem_driver = gdal.GetDriverByName("MEM")
    mem_raster = mem_driver.Create("", W, H, 1, gdal.GDT_Byte)
    mem_raster.SetGeoTransform(dataset.GetGeoTransform())
    mem_raster.SetProjection(dataset.GetProjection())
    gdal.RasterizeLayer(mem_raster, [1], layer, burn_values=[1])
    mask = mem_raster.ReadAsArray()

    # 提取训练样本
    y = mask.flatten()
    train_indices = y == 1
    X_train = X[train_indices]
    y_train = y[train_indices]

    if len(X_train) == 0:
        print("⚠️ 无监督分类样本，跳过")
        return np.zeros((H, W), dtype=np.uint8)

    # 标准化 + 分类
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_scaled = scaler.transform(X)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # 预测整图
    y_pred = clf.predict(X_scaled).reshape(H, W)
    return y_pred