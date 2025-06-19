# modules/clipper.py
import os
from osgeo import gdal

def clip_raster_gdal(input_raster, output_raster, shp_path):
    if os.path.exists(output_raster):
        print(f"⏩ 裁剪文件已存在，跳过: {output_raster}")
        return

    try:
        # 裁剪影像
        clipped_raster = gdal.Warp(output_raster, input_raster, cutlineDSName=shp_path, cropToCutline=True,
                                   dstNodata=-9999, warpOptions=["CUTLINE_ALL_TOUCHED=TRUE"])
        
        # 读取裁剪后的影像数据
        band = clipped_raster.GetRasterBand(1)
        data = band.ReadAsArray().astype(float)

        # 将数据值乘以 0.0001 来获取反射率
        data *= 0.0001

        # 创建输出文件并保存处理后的数据
        driver = gdal.GetDriverByName("GTiff")
        out_raster = driver.Create(output_raster, clipped_raster.RasterXSize, clipped_raster.RasterYSize, 1, gdal.GDT_Float32)

        # 设置地理信息
        out_raster.SetGeoTransform(clipped_raster.GetGeoTransform())
        out_raster.SetProjection(clipped_raster.GetProjection())

        # 写入数据
        out_band = out_raster.GetRasterBand(1)
        out_band.WriteArray(data)
        out_band.SetNoDataValue(-9999)

        # 释放资源
        out_band.FlushCache()
        out_band.ComputeStatistics(False)
        out_raster = None
        print(f"✅ 裁剪并保存处理后的影像: {output_raster}")
        
    except Exception as e:
        print(f"❌ 裁剪过程中出现错误: {e}")

def process_folders(data_dir, output_root, shapefile):
    print(f"🔍 开始处理文件夹: {data_dir}")
    
    os.makedirs(output_root, exist_ok=True)

    for root, dirs, files in os.walk(data_dir):
        if os.path.basename(root) == "clipped":
            continue

        tiff_files = [file for file in files if file.lower().endswith(".tif")]
        if not tiff_files:
            print(f"⚠️ 未找到 TIFF 文件，跳过: {root}")
            continue  

        folder_name = os.path.basename(root)
        output_folder = os.path.join(output_root, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        for tiff_file in tiff_files:
            tiff_file_path = os.path.join(root, tiff_file)
            clipped_file_path = os.path.join(output_folder, f"clipped_{tiff_file}")

            print(f"📌 处理文件: {tiff_file_path}")
            clip_raster_gdal(tiff_file_path, clipped_file_path, shapefile)
