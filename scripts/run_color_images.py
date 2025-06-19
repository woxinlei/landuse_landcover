import sys
import os

# 添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osgeo import gdal
from modules import image_processing
from modules import config  # 你存路径的配置文件
import numpy as np
def main(base_dir):
    output_dir = os.path.join(base_dir, "color_results")
    os.makedirs(output_dir, exist_ok=True)

    driver = gdal.GetDriverByName("GTiff")

    time_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    for folder in time_folders:
        clipped_dir = os.path.join(base_dir, folder)
        band4_path = os.path.join(clipped_dir, f"clipped_{folder}_band4.tif")
        band3_path = os.path.join(clipped_dir, f"clipped_{folder}_band3.tif")
        band2_path = os.path.join(clipped_dir, f"clipped_{folder}_band2.tif")
        band5_path = os.path.join(clipped_dir, f"clipped_{folder}_band5.tif")

        # 读取波段
        band4, ds4 = image_processing.read_band_with_nodata(band4_path)
        band3, ds3 = image_processing.read_band_with_nodata(band3_path)
        band2, ds2 = image_processing.read_band_with_nodata(band2_path)
        band5, ds5 = image_processing.read_band_with_nodata(band5_path)

        if any(b is None for b in (band4, band3, band2, band5)):
            print(f"⚠️ {folder} 缺少波段，跳过")
            continue

        # 真彩色
        true_color = np.dstack((band4, band3, band2))
        true_color_file = os.path.join(output_dir, f"true_color_{folder}.tif")
        if os.path.exists(true_color_file):
            print(f"⚠️ 已存在，跳过真彩色: {true_color_file}")
        else:
            image_processing.save_rgb_image(true_color_file, true_color, ds4.GetGeoTransform(), ds4.GetProjection(), driver)

        # 假彩色
        false_color = np.dstack((band5, band4, band3))
        false_color_file = os.path.join(output_dir, f"false_color_{folder}.tif")
        if os.path.exists(false_color_file):
            print(f"⚠️ 已存在，跳过假彩色: {false_color_file}")
        else:
            image_processing.save_rgb_image(false_color_file, false_color, ds4.GetGeoTransform(), ds4.GetProjection(), driver)

        print(f"✅ {folder} 真彩色和假彩色影像处理完成")

if __name__ == "__main__":
    base_dir = r"F:\data_eerduosi\data\data128032\dataoutput_clip"  # 你的数据根目录，按需修改
    main(base_dir)
