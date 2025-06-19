# modules/config.py
import os
dir =  r"F:\data_eerduosi\data\data128032"
# ---------------- 输入数据路径 ----------------
data_dir = os.path.join(dir, "LandsatL2C2") 

# ---------------- 输出数据根目录 ----------------
output_root = os.path.join(dir, "dataoutput_clip") 
output_root_folder =  os.path.join(dir, "landsat_output") 
base_dir = output_root
# ---------------- 矢量文件路径 ----------------
shp_dir =  r"F:\data_eerduosi\data\china_SHP"
shapefile = os.path.join(shp_dir, "OSM_eerduosi_128032.shp") 
shp_city = os.path.join(shp_dir, "gis_osm_landuse.shp") 
shp_water = os.path.join(shp_dir, "gis_osm_water.shp") 
shp_road = os.path.join(shp_dir, "gis_osm_road.shp") 

# ---------------- NDVI 和聚类相关 ----------------
ndvi_folder = os.path.join(base_dir, "ndvi_results")  # 你之前写的路径和这里保持一致
cluster_folder = os.path.join(base_dir, "supervised_kmeans_results_nocloud_withoutwater_city")

output_csv_ndvi_timetable = os.path.join(base_dir, "像素级_聚类_NDVI_时间表.csv")
output_csv_ndvi_timetable_cleaned = os.path.join(base_dir, "像素级_聚类_NDVI_时间表_cleaned.csv")

output_csv_crops = os.path.join(base_dir, "像素级聚类NDVI_地物分类_相对分配.csv")
output_csv_crops_correct = os.path.join(base_dir, "像素级_聚类_NDVI_地物分类_修正2.csv")
output_csv_crops_correct2 = os.path.join(base_dir, "像素级_聚类_NDVI_地物分类_修正2_农作物类型.csv")

# ---------------- 作物相关输出 ----------------
crops_folder = os.path.join(base_dir, "crops2")
output_csv_crops_correct2_abnormal = os.path.join(base_dir, "小麦_NDVI相对异常点_ratio9_neighbour3.csv")

# ---------------- 真彩色影像和动画 ----------------
true_color_dir = os.path.join(base_dir, "color_results")
output_gif_path = os.path.join(base_dir, "小麦_NDVI相对异常点_ratio8_neighbour3.gif")

# ---------------- 其他参数 ----------------
cloud_threshold = 22018  # 云阈值，可根据需要调整

# ---------------- 兼容老代码可能使用的路径变量 ----------------
output_dir_kmeans = cluster_folder
output_dir_ndvi = ndvi_folder