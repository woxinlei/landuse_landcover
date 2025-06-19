import sys
import os

# 添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# scripts/run_clipping.py
from modules import clipper
from modules import config

def main():
    clipper.process_folders(config.data_dir, config.output_root_folder, config.shapefile)

if __name__ == "__main__":
    main()
