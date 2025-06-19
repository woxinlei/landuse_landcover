import os
import numpy as np
from osgeo import gdal

def read_band_with_nodata(path):
    ds = gdal.Open(path)
    if ds is None:
        print(f"❌ 无法打开波段文件: {path}")
        return None, None
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, ds

def save_rgb_image(output_path, data_3d, geo_transform, projection, driver):
    data_3d = np.nan_to_num(data_3d, nan=0)
    ds = driver.Create(output_path, data_3d.shape[1], data_3d.shape[0], 3, gdal.GDT_Float32)
    if ds is None:
        print(f"❌ 无法创建文件 {output_path}")
        return False
    for i in range(3):
        band = ds.GetRasterBand(i+1)
        band.WriteArray(data_3d[:, :, i])
        band.SetNoDataValue(0)
        band.FlushCache()
    ds.SetGeoTransform(geo_transform)
    ds.SetProjection(projection)
    ds.FlushCache()
    return True
