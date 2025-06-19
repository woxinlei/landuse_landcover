import sys
import os

# 把项目根目录加入 sys.path，确保能找到 modules 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from modules import cloud_masking as cm
from modules import config

def main():
    cloud_threshold = config.cloud_threshold
    data_dir = config.data_dir
    output_root_folder = config.output_root_folder

    for root, dirs, files in os.walk(data_dir):
        mtl_files = [f for f in files if f.endswith("_MTL.txt")]
        if not mtl_files:
            continue

        folder_name = os.path.basename(root)
        output_folder = os.path.join(output_root_folder, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        for mtl_file in mtl_files:
            mtl_path = os.path.join(root, mtl_file)
            print(f"Processing {mtl_path}...")
            cm.process_image(mtl_path, cloud_threshold, root, output_folder)

if __name__ == "__main__":
    main()
