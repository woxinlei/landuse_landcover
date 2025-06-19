#NDVI
import os
import numpy as np
from osgeo import gdal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys

# 添加项目根目录到模块搜索路径，方便导入 modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import config
from modules import preprocessing, raster_utils, classification, postprocessing, io_utils 

#输出数据路径
output_dir_ndvi = os.path.join(config.base_dir, "ndvi_results")
os.makedirs(output_dir_ndvi, exist_ok=True)
# 获取所有 Landsat 影像的文件夹
time_folders = [folder for folder in os.listdir(config.base_dir) if os.path.isdir(os.path.join(config.base_dir, folder))]

for folder in time_folders:
    clipped_dir = os.path.join(config.base_dir, folder)

    # 读取波段数据
    bands = {}
    required_bands = [3, 4, 5, 6, 10]  # 重要波段
    try:
        for band in required_bands:
            bands[band], dataset = preprocessing.read_band(os.path.join(clipped_dir, f"clipped_{folder}_band{band}.tif"), 1)
            
    except:
        print(f"⚠️ {folder} 缺少必要波段，跳过")
        continue
    
    # 确保 0 值像素不参与计算
    zero_mask = (bands[3] == 0) | (bands[3] == -0.9999)  # 以 Band 3 为基准，假设 0或者0.9999值代表无效数据 -
    cloud_mask = (bands[3] == 0)
    out_mask = (bands[3] == -0.9999)
    # 计算指数
    ndvi, ndwi, ndbi = preprocessing.compute_indices(bands)


    # 栅格化城市和水体掩膜，排除非耕地
    city_mask = raster_utils.rasterize_shapefile(config.shp_city, dataset, burn_value=1)
    water_mask = raster_utils.rasterize_shapefile(config.shp_water, dataset, burn_value=1)
    road_mask = raster_utils.rasterize_shapefile(config.shp_road, dataset, burn_value=1)
    valid_mask = (city_mask == 0) & (water_mask == 0) & (road_mask == 0)


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

                features_scaled = scaler.transform(features)
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

    # 计算最终的 NDVI 掩膜（这里用前面算过的城市水体平滑掩膜）
    ndvi_masked = np.where(land_mask, ndvi, np.nan)
    # 计算 NDVI 的均值和标准差（忽略 NaN 值）
    ndvi_mean = np.nanmean(ndvi_masked)
    ndvi_std = np.nanstd(ndvi_masked)
    # Z-score 标准化
    ndvi_min = np.nanmin(ndvi_masked)
    ndvi_max = np.nanmax(ndvi_masked)
    #ndvi_standardized = (ndvi_masked - ndvi_mean) / (ndvi_std + eps)
    print(format(ndvi_min),format(ndvi_max))
    # 可视化 NDVI 结果
    plt.figure(figsize=(10, 6))
    plt.imshow(ndvi_masked, cmap='RdYlGn')
    plt.colorbar(label=" NDVI ")
    plt.title(f"{folder} ——NDVI")
    plt.show()
    
    
    output_path_ndvi = os.path.join(config.output_dir_ndvi, f"{folder}_ndvi_results.tif")
    io_utils.save_geotiff(output_path_ndvi,ndvi_masked,dataset,gdal.GDT_Float32,0)
