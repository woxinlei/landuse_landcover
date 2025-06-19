import os
import re
from osgeo import gdal
import numpy as np

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

    qa_match = re.search(r'FILE_NAME_QUALITY_L1_PIXEL\s=\s"([\w_]+QA_PIXEL\.TIF)"', "\n".join(lines))
    if qa_match:
        band_paths["QA"] = qa_match.group(1)
    return band_paths

def process_qa_band(qa_file_path):
    try:
        qa_dataset = gdal.Open(qa_file_path)
        if qa_dataset is None:
            print(f"错误: 无法打开 QA 文件 {qa_file_path}")
            return None
        qa_band = qa_dataset.GetRasterBand(1)
        qa_data = qa_band.ReadAsArray()
        return qa_data
    except Exception as e:
        print(f"处理 QA 文件 {qa_file_path} 时出错: {e}")
        return None

def count_qa_values(qa_data):
    unique, counts = np.unique(qa_data, return_counts=True)
    return dict(zip(unique, counts))
