import os
import re
from osgeo import gdal
import numpy as np

def read_band(dataset):
    band = dataset.GetRasterBand(1)
    return band.ReadAsArray()

def parse_mtl(mtl_path):
    with open(mtl_path, 'r') as f:
        lines = f.readlines()

    band_paths = {}
    for line in lines:
        match = re.search(r'FILE_NAME_BAND_(\d+)\s=\s"([\w_]+SR_B\d+\.TIF)"', line)
        if match:
            band_num = int(match.group(1))
            file_name = match.group(2)
            band_paths[band_num] = file_name

        thermal_match = re.search(r'FILE_NAME_BAND_ST_B10\s=\s"([\w_]+ST_B10\.TIF)"', line)
        if thermal_match:
            band_paths[10] = thermal_match.group(1)

    qa_match = re.search(r'FILE_NAME_QUALITY_L1_PIXEL\s=\s"([\w_]+QA_PIXEL\.TIF)"', "\n".join(lines))
    if qa_match:
        band_paths["QA"] = qa_match.group(1)
    else:
        print("Warning: QA_PIXEL band not found in MTL file.")
    return band_paths

def create_cloud_mask(qa_data, cloud_threshold=22018):
    return qa_data >= cloud_threshold

def save_band_data(output_file, band_data, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_file, band_data.shape[1], band_data.shape[0], 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(band_data)
    out_band.SetNoDataValue(-9999)
    out_ds.FlushCache()

def process_image(mtl_file_path, cloud_threshold, folder_path, output_folder):
    band_paths = parse_mtl(mtl_file_path)
    if "QA" not in band_paths:
        print("Error: QA band is missing.")
        return

    qa_file_path = os.path.join(folder_path, band_paths["QA"])
    qa_dataset = gdal.Open(qa_file_path)
    if qa_dataset is None:
        print(f"Error: Unable to open {qa_file_path}")
        return

    qa_data = read_band(qa_dataset)
    cloud_mask = create_cloud_mask(qa_data, cloud_threshold)

    for band_num in list(range(1, 8)) + [10]:
        band_path = band_paths.get(band_num)
        if band_path is None:
            print(f"Warning: Band {band_num} missing.")
            continue
        band_file_path = os.path.join(folder_path, band_path)
        band_dataset = gdal.Open(band_file_path)
        if band_dataset is None:
            print(f"Error: Unable to open {band_file_path}")
            continue
        band_data = read_band(band_dataset)

        band_data[cloud_mask] = -9999  # 用 NoData 值代替云区

        geo_transform = band_dataset.GetGeoTransform()
        projection = band_dataset.GetProjection()

        output_file = os.path.join(output_folder, f"{os.path.basename(folder_path)}_band{band_num}.tif")
        if os.path.exists(output_file):
            print(f"Skipped {output_file} (already exists).")
            continue
        save_band_data(output_file, band_data, geo_transform, projection)
        print(f"Saved {output_file}")
