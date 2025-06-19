import os
import sys
import numpy as np
from osgeo import gdal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import config
from modules import preprocessing, raster_utils, classification, postprocessing, io_utils

output_dir_kmeans = os.path.join(config.base_dir, "supervised_kmeans_results_nocloud_withoutwater_city")

os.makedirs(output_dir_kmeans, exist_ok=True)

# 获取所有 Landsat 影像的文件夹
time_folders = [folder for folder in os.listdir(config.base_dir) if os.path.isdir(os.path.join(config.base_dir, folder))]

for folder in time_folders:
    clipped_dir = os.path.join(config.base_dir, folder)

    # 读取波段数据
    bands = {}
    required_bands = [3, 4, 5, 6,10]  # 重要波段
    try:
        for band in required_bands:
            band_path = os.path.join(clipped_dir, f"clipped_{folder}_band{band}.tif")
            bands[band], dataset = preprocessing.read_band(band_path, 1)
    except FileNotFoundError as e:
            print(f"⚠️ 文件未找到: {e}, 跳过 {folder}")
            continue
    except Exception as e:
            print(f"⚠️ 读取波段出错: {e}, 跳过 {folder}")
            continue
    
    # 确保 0 值像素不参与计算
    zero_mask = (bands[3] == 0) | (bands[3] == -0.9999)  # 以 Band 3 为基准，假设 0或者0.9999值代表无效数据 -
    cloud_mask = (bands[3] == 0)
    out_mask = (bands[3] == -0.9999)
    # 计算指数
    ndvi, ndwi, ndbi = preprocessing.compute_indices(bands)

    # 栅格化矢量数据
    city_mask = raster_utils.rasterize_shapefile(config.shp_city, dataset, 1).astype(bool)  # 城市
    water_mask = raster_utils.rasterize_shapefile(config.shp_water, dataset, 1).astype(bool)  # 水体
    road_mask = raster_utils.rasterize_shapefile(config.shp_road, dataset, 1).astype(bool)  # 道路

    # 确保 0 值不参与计算
    city_mask &= ~zero_mask
    water_mask &= ~zero_mask
    road_mask &= ~zero_mask

    # 组合特征
    features = np.stack([bands[3], bands[4], bands[5], bands[6], ndvi, ndwi, ndbi], axis=-1)

    # 选择训练样本点
    H, W, C = features.shape
    features = features.reshape(-1, C)

    # 生成训练数据（监督分类）
    train_indices_city = np.where(city_mask.flatten())[0]  # 城市样本
    try:
        train_indices_non_city = np.random.choice(
            np.where(~city_mask.flatten() & ~water_mask.flatten() & ~road_mask.flatten() & ~zero_mask.flatten())[0],
            size=len(train_indices_city), replace=False
        )
    except ValueError:
        print(f"⚠️ 非城市样本不足，跳过 {folder}")
        continue

    # 城市分类
    if len(train_indices_city) == 0 or len(train_indices_non_city) == 0:
        print(f"⚠️ {folder} 中城市或非城市样本不足，跳过城市分类。")
        smoothed_city_mask = np.zeros((H, W), dtype=bool)
    else:
        try:
            # 组合训练样本
            X_train = np.vstack((features[train_indices_city], features[train_indices_non_city]))
            y_train = np.hstack((np.ones(len(train_indices_city)), np.zeros(len(train_indices_non_city))))  # 1=城市, 0=非城市

            # 标准化器
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            X_train_scaled = scaler.transform(X_train)

            # 训练模型
            model = classification.train_classifier(X_train_scaled, y_train, method="random_forest")

            # 预测整张图
            city_predictions = model.predict(features_scaled).reshape(H, W)

            # 合并已有城市掩膜
            final_city_mask = city_mask | (city_predictions == 1)
            final_city_mask &= ~zero_mask

            # 清洗 + 平滑
            city_mask_cleaned = postprocessing.remove_isolated_pixels(final_city_mask)
            city_mask_cleaned = postprocessing.remove_small_objects(city_mask_cleaned, min_size=50)

            final_city_mask = city_mask_cleaned | (city_predictions == 1)
            smoothed_city_mask = postprocessing.smooth_mask(final_city_mask, size=3)
            smoothed_city_mask &= ~zero_mask

        except Exception as e:
            print(f"⚠️ 城市分类失败于 {folder}：{e}")
            smoothed_city_mask = np.zeros((H, W), dtype=bool)
     # **水体滤波** (类似于城市的处理)
    
    # 生成训练数据（监督分类）
    train_indices_water = np.where(water_mask.flatten())[0]

    try:
        train_indices_non_water = np.random.choice(
            np.where(~city_mask.flatten() & ~water_mask.flatten() & ~road_mask.flatten() & ~zero_mask.flatten())[0],
            size=len(train_indices_water), replace=False
        )
    except ValueError:
        print(f"⚠️ {folder} 中非水体样本不足，跳过水体分类。")
        smoothed_water_mask = np.zeros((H, W), dtype=bool)
    else:
        if len(train_indices_water) == 0 or len(train_indices_non_water) == 0:
            print(f"⚠️ {folder} 中水体或非水体样本不足，跳过水体分类。")
            smoothed_water_mask = np.zeros((H, W), dtype=bool)
        else:
            try:
                X_train_water = np.vstack((features[train_indices_water], features[train_indices_non_water]))
                y_train_water = np.hstack((np.ones(len(train_indices_water)), np.zeros(len(train_indices_non_water))))

                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                X_train_water = scaler.transform(X_train_water)

                model_water = classification.train_classifier(X_train_water, y_train_water, method="random_forest")

                water_predictions = model_water.predict(features_scaled).reshape(H, W)

                final_water_mask = water_mask | (water_predictions == 1)
                final_water_mask &= ~zero_mask

                water_mask_cleaned = postprocessing.remove_isolated_pixels(final_water_mask)
                water_mask_cleaned = postprocessing.remove_small_objects(water_mask_cleaned, min_size=50)

                final_water_mask = water_mask_cleaned | (water_predictions == 1)
                smoothed_water_mask = postprocessing.smooth_mask(final_water_mask, size=3)
                smoothed_water_mask &= ~zero_mask

            except Exception as e:
                print(f"⚠️ {folder} 中水体模型训练或预测失败: {e}")
                smoothed_water_mask = np.zeros((H, W), dtype=bool)


    # 计算最终的土地掩膜，去除城市、道路、水体，同时保留 0 值不变
    land_mask = ~smoothed_city_mask & ~smoothed_water_mask & ~road_mask & ~zero_mask
    # 继续去除城市区域，并进行 K-Means
    land_features = features[land_mask.flatten()]

    # 进行 K-Means 分类
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = np.full((H, W), -1)  # 先创建一个默认填充的矩阵

    # 仅对陆地区域进行分类
    predicted_labels = kmeans.fit_predict(land_features)

    # **1. 统计每个类别的像素数量**
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)

    # **2. 按像素数从大到小排序**
    sorted_indices = np.argsort(-counts)  # 降序排序
    remap_dict = {unique_labels[idx]: len(unique_labels) - i for i, idx in enumerate(sorted_indices)}

    # 重新映射类别编号
    sorted_labels = np.vectorize(remap_dict.get)(predicted_labels)

    # 赋值回原始矩阵
    kmeans_labels = np.full((H, W), -1)  # 先填充默认值
    kmeans_labels[land_mask] = sorted_labels  # 仅赋值 land_mask 位置

 
    kmeans_labels_cleaned = postprocessing.remove_small_clusters(kmeans_labels, min_size=5)

    
    output_path_landuse = os.path.join(output_dir_kmeans, f"{folder}_supervised_city.tif")
    io_utils.save_geotiff(output_path_landuse,kmeans_labels_cleaned,dataset,gdal.GDT_Float32,0)

    
    
    # 组合最终结果
    final_labels = np.full((H, W), -1)
    final_labels[smoothed_city_mask] = 99  # 城市区域
    final_labels[smoothed_water_mask] = 100  # 水体
    final_labels[road_mask] = 101  # 道路
    final_labels[land_mask] = kmeans_labels_cleaned[land_mask] 
    
    output_path_water_city = os.path.join(output_dir_kmeans, f"{folder}_supervised_kmeans_no_city.tif")
    io_utils.save_geotiff(output_path_water_city,final_labels,dataset,gdal.GDT_Float32,-1)