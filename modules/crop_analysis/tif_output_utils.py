import os
import numpy as np
from osgeo import gdal, ogr, osr
import pandas as pd

def apply_crop_labels_to_tif(csv_path, tif_folder, output_folder):
    """
    将CSV中的地物/作物类型标签应用到对应的TIF图像上，输出新的分类TIF文件。
    """
    os.makedirs(output_folder, exist_ok=True)

    # 读取CSV
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date_str'] = df['timestamp'].dt.strftime('%Y%m%d')
    data_map = df.set_index(['date_str', 'row', 'col'])['corrected_type'].to_dict()

    # 定义分类映射（可扩展）
    class_mapping = {
        '林地': 1,
        '耕地': 2,
        '待耕地': 3,
        '裸地': 4,
        '小麦': 5,
        '水稻': 6
        #后续作物
    }

    tif_files = [f for f in os.listdir(tif_folder) if f.endswith("_supervised_kmeans_no_city.tif")]
    total = len(tif_files)
    print(f"📦 共找到 {total} 个 TIF 文件，开始处理...")

    for i, filename in enumerate(tif_files, 1):
        parts = filename.split("_")
        if len(parts) < 4:
            print(f"⚠️ 跳过不符合命名规范的文件：{filename}")
            continue

        date_str = parts[3]
        tif_path = os.path.join(tif_folder, filename)

        dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)
        if dataset is None:
            print(f"❌ 无法读取文件：{tif_path}")
            continue

        band = dataset.GetRasterBand(1)
        arr = band.ReadAsArray()
        modified = np.copy(arr)
        rows, cols = arr.shape

        update_count = 0
        for row in range(rows):
            for col in range(cols):
                key = (date_str, row, col)
                label = data_map.get(key)
                if label and label in class_mapping:
                    modified[row, col] = class_mapping[label]
                    update_count += 1

        # 输出路径和写入文件
        driver = gdal.GetDriverByName("GTiff")
        prefix = filename.replace("_supervised_kmeans_no_city.tif", "")
        out_filename = prefix + "_crop.tif"
        out_path = os.path.join(output_folder, out_filename)

        out_ds = driver.Create(out_path, cols, rows, 1, band.DataType)
        out_ds.SetGeoTransform(dataset.GetGeoTransform())
        out_ds.SetProjection(dataset.GetProjection())
        out_ds.GetRasterBand(1).SetNoDataValue(0)
        out_ds.GetRasterBand(1).WriteArray(modified)
        out_ds.FlushCache()

        # 清理
        out_ds = None
        dataset = None

        print(f"✅ [{i}/{total}] 写入完成：{out_filename}（覆盖像素数: {update_count}）")    
def vectorize_tif_by_class(input_folder, output_subdir=True):
    """
    将 input_folder 中每个分类 TIF 文件的每个类别单独矢量化为 Shapefile。
    
    参数：
        input_folder (str): 包含.tif文件的目录
        output_subdir (bool): 是否将每个tif结果存储在独立子文件夹中
    """
    tif_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]
    total = len(tif_files)
    print(f"📦 共找到 {total} 个 TIF 文件，开始矢量化...")

    for idx, tif_file in enumerate(tif_files, 1):
        tif_path = os.path.join(input_folder, tif_file)
        print(f"\n📂 [{idx}/{total}] 正在处理: {tif_path}")

        src_ds = gdal.Open(tif_path)
        if src_ds is None:
            print(f"❌ 无法打开 {tif_path}")
            continue

        band = src_ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        unique_values = set(np.unique(arr))
        if nodata is not None:
            unique_values.discard(nodata)

        # 输出目录
        base_name = os.path.splitext(tif_file)[0]
        output_dir = os.path.join(input_folder, base_name) if output_subdir else input_folder
        os.makedirs(output_dir, exist_ok=True)

        # 获取地理信息
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_ds.GetProjection())
        geotransform = src_ds.GetGeoTransform()

        for val in sorted(unique_values):
            print(f"▶️  矢量化类别: {val}")

            # 创建临时掩膜
            mask_arr = np.where(arr == val, 1, 0).astype(np.uint8)
            mem_driver = gdal.GetDriverByName('MEM')
            mask_ds = mem_driver.Create('', src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Byte)
            mask_ds.SetGeoTransform(geotransform)
            mask_ds.SetProjection(src_ds.GetProjection())
            mask_ds.GetRasterBand(1).WriteArray(mask_arr)

            # 输出 shapefile
            shp_path = os.path.join(output_dir, f"class_{val}.shp")
            drv = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(shp_path):
                drv.DeleteDataSource(shp_path)
            out_ds = drv.CreateDataSource(shp_path)
            out_layer = out_ds.CreateLayer(f"class_{val}", srs=srs)
            out_layer.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))

            gdal.Polygonize(
                mask_ds.GetRasterBand(1),
                mask_ds.GetRasterBand(1),
                out_layer,
                0,
                [],
                callback=None
            )

            print(f"✅ 已保存: {shp_path}")
            mask_ds = None
            out_ds = None

        src_ds = None