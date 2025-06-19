from osgeo import gdal
import os
import numpy as np

def save_geotiff(output_path, array, reference_dataset, data_type=gdal.GDT_Byte, nodata_val=None):
    """
    保存 numpy array 为 GeoTIFF 文件，参考一个已有数据集的投影和地理变换信息。

    参数:
    - output_path: 输出文件路径
    - array: 2D numpy 数组
    - reference_dataset: GDAL Dataset，参考的影像，用于获取投影和仿射变换
    - data_type: GDAL 数据类型，默认字节型
    - nodata_val: 可选，指定 NoData 值（如 0 或 -9999），默认不设置
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    H, W = array.shape

    driver = gdal.GetDriverByName("GTiff")
    out_raster = driver.Create(output_path, W, H, 1, data_type)
    out_raster.SetGeoTransform(reference_dataset.GetGeoTransform())
    out_raster.SetProjection(reference_dataset.GetProjection())

    band = out_raster.GetRasterBand(1)
    if nodata_val is not None:
        band.SetNoDataValue(nodata_val)
        # 将 NaN 替换成 nodata_val，避免写入文件时出错
        array_to_write = np.where(np.isnan(array), nodata_val, array)
    else:
        array_to_write = array

    band.WriteArray(array_to_write)
    out_raster.FlushCache()
    out_raster = None
    
    print(f"✅ 文件保存成功: {output_path}")