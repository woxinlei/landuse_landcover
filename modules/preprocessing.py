# modules/preprocessing.py
import numpy as np
from osgeo import gdal

def read_band(file_path, band_number=1):
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"文件不存在或无法打开: {file_path}")
    band = dataset.GetRasterBand(band_number)
    if band is None:
        raise ValueError(f"影像中不存在波段 {band_number}")
    return band.ReadAsArray().astype(np.float32), dataset

def compute_indices(bands, eps=1e-6):
    ndvi = (bands[5] - bands[4]) / (bands[5] + bands[4] + eps)
    ndwi = (bands[5] - bands[3]) / (bands[5] + bands[3] + eps)
    ndbi = (bands[6] - bands[5]) / (bands[6] + bands[5] + eps)
    return ndvi, ndwi, ndbi
