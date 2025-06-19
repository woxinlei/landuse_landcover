# 土地分类与作物识别项目

基于 Landsat 影像，完成鄂尔多斯地区的土地覆盖分类、NDVI时间序列分析、作物类型识别及异常点检测。项目采用模块化设计，方便维护与复用。

---

## 项目结构

F:
│ README.md
│ run_all_pipeline.py # 统一运行所有子流程的脚本
│ structure.txt
│
├─modules # 核心功能模块
│ │ config.py # 全局路径及参数配置
│ │ qa_utils.py # 影像质量统计相关函数
│ │ cloud_masking.py # 云掩膜处理模块
│ │ clipper.py # 影像裁剪模块
│ │ image_processing.py # 计算NDVI等指数
│ │ preprocessing.py # 数据预处理函数
│ │ raster_utils.py # 栅格处理工具
│ │ classification.py # 监督分类相关函数
│ │ clustering.py # 聚类分析
│ │ postprocessing.py # 结果平滑及去噪
│ │ io_utils.py # 输入输出辅助
│ │ init.py
│ │
│ └─crop_analysis # 作物识别与异常检测子模块
│ │ anomaly_detection.py # 作物NDVI异常检测与动画
│ │ tif_output_utils.py # 作物分类写入TIF及矢量化
│ │ landtype_analysis.py # NDVI聚类数据提取与分类
│ │ init.py
│
└─scripts # 主要流程脚本
│ run_qa_count.py # 质量统计
│ run_cloud_masking.py # 云掩膜制作
│ run_clipping.py # 影像裁剪
│ run_color_images.py # 真彩色合成影像
│ run_landuse_pipeline.py # 土地覆盖监督分类与聚类
│ run_ndvi_pipeline.py # NDVI计算与保存
│ run_crop_landtype_pipeline.py # 作物识别与异常检测


---

## 功能概述

- **run_qa_count.py**  
  统计影像的质量和云量，为后续处理提供依据。
  
- **run_cloud_masking.py**  
  基于质量控制文件制作云掩膜，去除云干扰区域。

- **run_clipping.py**  
  根据矢量边界裁剪影像，聚焦研究区域。

- **run_color_images.py**  
  生成真彩色影像，便于人工目视检查。

- **run_landuse_pipeline.py**  
  土地覆盖的监督分类（城市、水体掩膜）和基于指数的KMeans聚类。

- **run_ndvi_pipeline.py**  
  计算NDVI、NDWI、NDBI等植被指数，保存时序数据。

- **run_crop_landtype_pipeline.py**  
  基于聚类和NDVI时序数据识别作物类型，检测NDVI异常点，生成作物分类影像和异常点动画。

---

## 依赖环境

- Python 3.9+
- 依赖库（可通过`requirements.txt`安装）：
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - scipy
  - GDAL


```bash
pip install -r requirements.txt
配置说明
所有路径和参数配置集中在 modules/config.py，请根据你的数据存放位置和需求修改配置。

运行说明
你可以逐个运行 scripts/ 下的脚本，按照推荐的顺序：

bash
复制
编辑
python scripts/run_qa_count.py
python scripts/run_cloud_masking.py
python scripts/run_clipping.py
python scripts/run_color_images.py
python scripts/run_landuse_pipeline.py
python scripts/run_ndvi_pipeline.py
python scripts/run_crop_landtype_pipeline.py
或者使用项目根目录下的 run_all_pipeline.py，自动顺序执行：

bash
复制
编辑
python run_all_pipeline.py
该脚本依次调用上述脚本，遇到错误会自动停止并打印错误信息。

常见问题
找不到模块 modules
确保在项目根目录运行脚本，或者在脚本中正确设置 sys.path。

编码错误 (UnicodeDecodeError)
脚本运行时涉及命令行输出时，请设置合适的编码（如utf-8）以避免编码错误。

路径问题
尽量使用 os.path.join 组织路径，避免手动拼写绝对路径。

开发建议
代码功能模块化，便于维护和调试。

版本管理请使用 Git，忽略大数据文件和缓存。

建议在虚拟环境中管理依赖。

联系方式
如有疑问或建议，欢迎联系项目负责人或提交Issue。

woxinlei
2025年6月