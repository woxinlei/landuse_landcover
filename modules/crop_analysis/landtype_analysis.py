import os
import numpy as np
from osgeo import gdal
import pandas as pd

def extract_pixel_ndvi_cluster(ndvi_folder, cluster_folder, output_csv_path):
    """
    从 NDVI 和聚类 TIF 图中提取每个像素的聚类 ID、NDVI 值和时间戳，并保存为 CSV。

    参数:
    - ndvi_folder: 存放 NDVI 结果的文件夹路径
    - cluster_folder: 存放对应聚类图的文件夹路径
    - output_csv_path: 输出 CSV 文件完整路径
    """
    ndvi_files = [
        os.path.join(ndvi_folder, f)
        for f in os.listdir(ndvi_folder)
        if f.endswith("_ndvi_results.tif")
    ]

    all_rows = []

    for ndvi_fp in ndvi_files:
        basename = os.path.basename(ndvi_fp)
        parts = basename.split("_")
        if len(parts) < 7:
            continue
        timestamp = f"{parts[3]}_{parts[4]}"

        cluster_name = basename.replace("_ndvi_results.tif", "_supervised_kmeans_no_city.tif")
        cluster_fp = os.path.join(cluster_folder, cluster_name)

        if not os.path.exists(cluster_fp):
            print(f"⚠️ 跳过未找到聚类文件: {cluster_fp}")
            continue

        ndvi_ds = gdal.Open(ndvi_fp)
        cluster_ds = gdal.Open(cluster_fp)
        ndvi = ndvi_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        cluster = cluster_ds.GetRasterBand(1).ReadAsArray().astype(np.int32)

        rows, cols = ndvi.shape
        for i in range(rows):
            for j in range(cols):
                c = cluster[i, j]
                v = ndvi[i, j]
                if c == 0 or not (-1 <= v <= 1):  # 过滤无效值
                    continue
                all_rows.append({
                    "row": i,
                    "col": j,
                    "cluster_id": c,
                    "timestamp": timestamp,
                    "ndvi_value": v
                })

    # 保存为 DataFrame
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ 已保存像素级聚类-NDVI表格，共记录条数：{len(df)}")

def clean_cluster_data(input_csv_path, output_csv_path, valid_clusters=[1, 2, 3, 4]):
    """
    清洗像素级聚类 NDVI 数据，只保留指定 cluster_id 的记录。

    参数:
    - input_csv_path: 输入的原始 CSV 文件路径
    - output_csv_path: 清洗后 CSV 文件的保存路径
    - valid_clusters: 保留的 cluster_id 列表，默认是 [1, 2, 3, 4]
    """
    if not os.path.exists(input_csv_path):
        print(f"❌ 找不到输入文件: {input_csv_path}")
        return

    # 读取数据
    df = pd.read_csv(input_csv_path)

    # 只保留指定 cluster_id 的记录
    df_cleaned = df[df['cluster_id'].isin(valid_clusters)]

    # 保存结果
    df_cleaned.to_csv(output_csv_path, index=False)
    print(f"✅ 已完成清洗，仅保留 cluster_id 为 {valid_clusters} 的数据，共 {len(df_cleaned)} 条记录")
# ✅ 函数应放在外部，供主函数调用
def assign_land_type(group):
    ranked = group.sort_values(by='ndvi_mean', ascending=False).copy()
    initial_land_types = ['林地', '耕地', '待耕地', '裸地']
    ranked['land_type'] = initial_land_types[:len(ranked)]

    idx = ranked.index.tolist()

    # 📌 自定义规则
    if ranked.loc[idx[0], 'ndvi_mean'] < 0.2:
        ranked.loc[idx[0], 'land_type'] = '待耕地'
    if ranked.loc[idx[1], 'ndvi_mean'] < 0.2:
        ranked.loc[idx[1], 'land_type'] = '待耕地'
    if ranked.loc[idx[2], 'ndvi_mean'] > 0.25:
        ranked.loc[idx[2], 'land_type'] = '耕地'
    if ranked.loc[idx[3], 'ndvi_mean'] > 0.25:
        ranked.loc[idx[3], 'land_type'] = '耕地'

    print(f"\n📅 Timestamp: {group['timestamp'].iloc[0]}")
    print(ranked[['cluster_id', 'ndvi_mean', 'ndvi_max', 'ndvi_min', 'ndvi_std', 'ndvi_amp', 'ndvi_median', 'land_type']])

    return ranked

# ✅ 主函数
def classify_land_types_from_ndvi(csv_input_path, csv_output_path):
    df = pd.read_csv(csv_input_path)
    df['date'] = pd.to_datetime(df['timestamp'].str.split('_').str[0], format="%Y%m%d")

    grouped = df.groupby(['timestamp', 'cluster_id'])['ndvi_value'].agg(
        ndvi_median='median', 
        ndvi_mean='mean',
        ndvi_max='max',
        ndvi_min='min',
        ndvi_std='std',
        ndvi_amp=lambda x: x.max() - x.min()
    ).reset_index()

    grouped_labeled = grouped.groupby('timestamp').apply(assign_land_type).reset_index(drop=True)

    df = df.merge(grouped_labeled[['timestamp', 'cluster_id', 'land_type']],
                  on=['timestamp', 'cluster_id'], how='left')

    df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 地物分类完成，结果保存至：{csv_output_path}")

def correct_land_type_sequence(land_types):
    """按规则修正时间序列中的地物类型"""
    bare_like = {'裸地', '待耕地'}
    veg_like = {'耕地', '林地'}

    corrected_types = []
    has_bare = any(lt in bare_like for lt in land_types)
    has_veg = any(lt in veg_like for lt in land_types)

    # 如果全是裸地或待耕地，则全部设为裸地
    if all(lt in bare_like for lt in land_types):
        return ['裸地'] * len(land_types)

    for lt in land_types:
        if has_bare and lt == '林地':
            corrected_types.append('耕地')
        elif has_veg and lt in bare_like:
            corrected_types.append('待耕地')
        else:
            corrected_types.append(lt)
    return corrected_types

def apply_land_type_correction(input_path, output_path):
    
    """读取像素级NDVI地物类型，按时间序列进行地物修正"""

    print("📥 正在读取原始数据...")
    df = pd.read_csv(input_path)

    # 只保留我们关心的类别
    df = df[df['land_type'].isin(['林地', '耕地', '裸地', '待耕地'])]

    # 转换为时间戳便于排序
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.split("_").str[0], format="%Y%m%d")

    # 如果输出文件存在则删除
    if os.path.exists(output_path):
        os.remove(output_path)

    # 写入表头
    df.head(0).assign(corrected_type='').to_csv(output_path, index=False, encoding='utf-8-sig')

    grouped = df.groupby(['row', 'col'])
    total = len(grouped)
    print(f"🧩 共需处理像素点：{total}")
    count = 0

    for (row, col), group in grouped:
        group = group.sort_values("timestamp")
        land_types = list(group['land_type'])

        corrected = correct_land_type_sequence(land_types)
        group['corrected_type'] = corrected

        # 追加写入
        group.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8-sig')

        count += 1
        if count % 1000 == 0 or count == total:
            print(f"✅ 已处理 {count}/{total} 个像素点...")

    print("\n🎉 修正完成，结果已保存至：", output_path)
def classify_crop_types(input_path, output_path):
    """根据时间序列 NDVI 类别进行作物/林地修正分类"""

    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[['row', 'col', 'timestamp', 'corrected_type', 'ndvi_value']]
    df['month'] = df['timestamp'].dt.month

    grouped = df.groupby(['row', 'col'])

    if os.path.exists(output_path):
        os.remove(output_path)

    first_write = True
    total = len(grouped)

    for i, ((row, col), group) in enumerate(grouped):
        if i % 1000 == 0 or i == total - 1:
            print(f"📍 正在处理第 {i+1}/{total} 个像素点：row={row}, col={col}")

        group = group.sort_values('timestamp').copy()
        types = group['corrected_type'].tolist()
        months = group['month'].tolist()
        timestamps = group['timestamp'].tolist()

        # === Step 1：林地判断 ===
        growing_season = [(4 <= m <= 9) for m in months]
        types_in_growing = [t for t, cond in zip(types, growing_season) if cond]
        is_forest = (
            any(t in {'耕地', '林地'} for t in types_in_growing) and
            all(t not in {'裸地', '待耕地'} for t in types_in_growing)
        )
        if is_forest:
            types = ['林地' if t == '耕地' else t for t in types]

        # === Step 2：作物分类 ===
        wheat_cutoff = rice_cutoff = second_wheat_start_ts = None

        for lt, m, ts in zip(types, months, timestamps):
            if lt in {'裸地', '待耕地'} and m in {5, 6} and wheat_cutoff is None:
                wheat_cutoff = ts
            if lt in {'裸地', '待耕地'} and m in {9, 10} and rice_cutoff is None:
                rice_cutoff = ts
        for lt, m, ts in zip(types, months, timestamps):
            if lt in {'裸地', '待耕地'} and m in {9, 10, 11}:
                second_wheat_start_ts = ts
                break

        new_types = []
        for lt, m, ts in zip(types, months, timestamps):
            if lt == '耕地':
                if wheat_cutoff and ts < wheat_cutoff and m <= 6:
                    new_types.append('小麦')
                elif rice_cutoff and ts < rice_cutoff and 7 <= m <= 10:
                    new_types.append('水稻')
                elif second_wheat_start_ts and ts > second_wheat_start_ts:
                    new_types.append('小麦')
                else:
                    new_types.append('耕地')
            else:
                new_types.append(lt)

        group['corrected_type'] = new_types
        group.to_csv(output_path, mode='a', index=False, encoding='utf-8-sig', header=first_write)
        first_write = False

    print("\n🎯 作物与林地类型已识别完毕，输出文件位于：", output_path)

