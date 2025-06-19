import os
import sys

# 先把根目录加进sys.path，保证import能找到modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import config
from modules.crop_analysis.landtype_analysis import (
    extract_pixel_ndvi_cluster,
    clean_cluster_data,
    classify_land_types_from_ndvi,
    apply_land_type_correction,
    classify_crop_types,
)
from modules.crop_analysis.tif_output_utils import (
    apply_crop_labels_to_tif,
    vectorize_tif_by_class,
)
from modules.crop_analysis.anomaly_detection import (
    detect_wheat_ndvi_anomalies,
    plot_ndvi_anomaly_animation,
)

def main():
    # 从 config 中读取路径配置
    ndvi_folder = config.ndvi_folder
    cluster_folder = config.cluster_folder
    crops_folder = config.crops_folder
    true_color_dir = config.true_color_dir

    output_csv_ndvi_timetable = config.output_csv_ndvi_timetable
    output_csv_ndvi_timetable_cleaned = config.output_csv_ndvi_timetable_cleaned
    output_csv_crops = config.output_csv_crops
    output_csv_crops_correct = config.output_csv_crops_correct
    output_csv_crops_correct2 = config.output_csv_crops_correct2
    output_csv_crops_correct2_abnormal = config.output_csv_crops_correct2_abnormal
    output_gif_path = config.output_gif_path

    os.makedirs(crops_folder, exist_ok=True)

    # --- 依次执行各步骤 ---
    extract_pixel_ndvi_cluster(ndvi_folder, cluster_folder, output_csv_ndvi_timetable)
    clean_cluster_data(output_csv_ndvi_timetable, output_csv_ndvi_timetable_cleaned)
    classify_land_types_from_ndvi(output_csv_ndvi_timetable_cleaned, output_csv_crops)
    apply_land_type_correction(output_csv_crops, output_csv_crops_correct)
    classify_crop_types(output_csv_crops_correct, output_csv_crops_correct2)
    apply_crop_labels_to_tif(output_csv_crops_correct2, cluster_folder, crops_folder)
    vectorize_tif_by_class(crops_folder)

    detect_wheat_ndvi_anomalies(
        output_csv_crops_correct2,
        output_csv_crops_correct2_abnormal,
        relative_drop_ratio=0.9,
        neighbor_radius=1,
        min_neighbor_count=3
    )
    display = plot_ndvi_anomaly_animation(
        output_csv_crops_correct2_abnormal,
        true_color_dir,
        output_gif_path
    )

    # 如果在Jupyter中运行，可以显示动画
    # from IPython.display import display as ipy_display
    # ipy_display(display)

if __name__ == "__main__":
    main()
