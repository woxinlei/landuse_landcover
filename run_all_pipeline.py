import subprocess
import os

# 设置脚本目录
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), 'scripts')

# 定义要按顺序执行的脚本名
scripts_to_run = [
    "run_qa_count.py",
    "run_cloud_masking.py",
    "run_clipping.py",
    "run_color_images.py",
    "run_landuse_pipeline.py",
    "run_ndvi_pipeline.py",
    "run_crop_landtype_pipeline.py",  # 放在最后
]

def run_pipeline():
    for script in scripts_to_run:
        script_path = os.path.join(SCRIPT_DIR, script)
        print(f"\n🚀 正在运行：{script} ...")
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',   # 指定utf-8解码输出
            errors='ignore'     # 忽略解码错误
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ 脚本 {script} 运行失败！错误信息：")
            print(result.stderr)
            break
        else:
            print(f"✅ {script} 完成")

if __name__ == "__main__":
    run_pipeline()