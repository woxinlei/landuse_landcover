import sys
import os

# 添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
from tqdm import tqdm
from modules import qa_utils
from modules import config  # 你存路径的配置文件

def main():
    qa_value_counts = {}

    print(f"扫描目录: {config.data_dir}")

    for root, dirs, files in os.walk(config.data_dir):
        print(f"检查目录: {root}")
        mtl_files = [file for file in files if file.endswith("_MTL.txt")]
        if not mtl_files:
            print(f"在 {root} 中未找到 MTL 文件")
        for mtl_file in tqdm(mtl_files, desc="处理 MTL 文件", unit="文件"):
            mtl_path = os.path.join(root, mtl_file)
            print(f"正在处理 MTL 文件: {mtl_path}")

            band_paths = qa_utils.parse_mtl(mtl_path)

            if "QA" in band_paths:
                qa_file_path = os.path.join(root, band_paths["QA"])
                print(f"找到 QA 波段文件: {qa_file_path}")

                qa_data = qa_utils.process_qa_band(qa_file_path)
                if qa_data is not None:
                    value_counts = qa_utils.count_qa_values(qa_data)
                    qa_value_counts[qa_file_path] = value_counts
                    print(f"{qa_file_path} 的 QA 值计数: {value_counts}")
                else:
                    print(f"由于错误，跳过 QA 文件 {qa_file_path}")
            else:
                print(f"{mtl_path} 中未找到 QA 波段")

    print("\n最终的 QA 值计数：")
    for qa_file_path, value_counts in tqdm(qa_value_counts.items(), desc="处理 QA 值计数", unit="文件"):
        print(f"QA 文件: {qa_file_path}")
        for value, count in value_counts.items():
            print(f"  值 {value}: {count} 次")

if __name__ == "__main__":
    main()
