import os
import numpy as np
from osgeo import gdal
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib import rcParams
def detect_wheat_ndvi_anomalies(input_csv,
                                 output_csv,
                                 crop_type='小麦',
                                 relative_drop_ratio=0.9,
                                 neighbor_radius=1,
                                 min_neighbor_count=3,
                                 verbose=True):
    """
    检测指定农作物类型在NDVI上的相对异常点（邻域下降）并保存结果。
    
    参数：
        input_csv (str): 输入CSV路径，包含列['row', 'col', 'timestamp', 'ndvi_value', 'corrected_type']
        output_csv (str): 输出CSV路径
        crop_type (str): 要检测的农作物类型（默认 '小麦'）
        relative_drop_ratio (float): 当前像素NDVI小于邻域中位数的百分比阈值
        neighbor_radius (int): 邻域搜索的像素半径
        min_neighbor_count (int): 至少需要的邻居数量才做判断
        verbose (bool): 是否打印进度日志
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"❌ 输入文件不存在：{input_csv}")

    df = pd.read_csv(input_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['corrected_type'] == crop_type].copy()

    if verbose:
        print(f"✅ 筛选作物类型: {crop_type}，样本数：{len(df)}")

    coords = df[['row', 'col']].drop_duplicates().to_numpy()
    tree = cKDTree(coords)
    if verbose:
        print(f"✅ 建立空间索引，共 {len(coords)} 个像素点")

    ndvi_map = df.set_index(['row', 'col', 'timestamp'])['ndvi_value']

    if os.path.exists(output_csv):
        os.remove(output_csv)

    first_write = True
    count_anomaly = 0

    for idx, ((r, c), group) in enumerate(df.groupby(['row', 'col'])):
        neighbor_idx = tree.query_ball_point([r, c], r=neighbor_radius)
        neighbor_coords = [tuple(coords[j]) for j in neighbor_idx if (coords[j][0], coords[j][1]) != (r, c)]
        if not neighbor_coords:
            continue

        for _, row_data in group.iterrows():
            t = row_data['timestamp']
            ndvi_val = row_data['ndvi_value']

            neighbor_ndvis = []
            for nr, nc in neighbor_coords:
                try:
                    neighbor_ndvis.append(ndvi_map.loc[(nr, nc, t)])
                except KeyError:
                    continue

            if len(neighbor_ndvis) < min_neighbor_count:
                continue

            median_val = np.median(neighbor_ndvis)
            ndvi_ratio = ndvi_val / median_val if median_val > 0 else 0
            ndvi_diff = ndvi_val - median_val

            if ndvi_ratio < relative_drop_ratio:
                record = {
                    'row': r,
                    'col': c,
                    'timestamp': t,
                    'ndvi_value': ndvi_val,
                    'median_neighbor_ndvi': median_val,
                    'ndvi_ratio': ndvi_ratio,
                    'ndvi_diff': ndvi_diff
                }
                pd.DataFrame([record]).to_csv(output_csv, mode='w' if first_write else 'a',
                                              header=first_write, index=False, encoding='utf-8-sig')
                first_write = False
                count_anomaly += 1

        if verbose and idx % 5000 == 0:
            print(f"👉 进度：{idx}个像素点，当前=({r}, {c})")

    print(f"✅ 检测完成，共检测到 {count_anomaly} 个异常点，结果保存至：{output_csv}")
def plot_ndvi_anomaly_animation(csv_path,
                               true_color_dir,
                               output_gif_path,
                               clip_min=0.9,
                               clip_max=1.4,
                               gamma=0.8,
                               point_size=0.5,
                               alpha=0.5,
                               interval=500,
                               fps=1,
                               show_jshtml=True):
    """
    制作NDVI异常点时间序列动画（叠加真彩色影像和异常点散点图）
    
    参数:
    - csv_path: 异常点CSV文件路径，包含 'timestamp', 'row', 'col' 等列
    - true_color_dir: 真彩色TIF影像文件夹，文件名需包含日期（格式yyyyMMdd）
    - output_gif_path: 输出GIF动画路径
    - clip_min, clip_max, gamma: 色彩拉伸和Gamma校正参数
    - point_size: 散点大小
    - alpha: 散点透明度
    - interval: 动画帧间隔（毫秒）
    - fps: 输出GIF帧率
    - show_jshtml: 是否返回JSHTML用于Notebook中直接展示
    """

    # 设置中文字体及负号正常显示
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df['date'] = df['timestamp'].dt.date
    grouped = df.groupby('date')
    dates = list(grouped.groups.keys())

    # 读取真彩色影像并做拉伸
    def fixed_clip_stretch(img, clip_min, clip_max, gamma):
        img_out = np.empty_like(img, dtype=np.uint8)
        for i in range(img.shape[2]):
            band = img[:, :, i]
            band_clip = np.clip(band, clip_min, clip_max)
            band_norm = (band_clip - clip_min) / (clip_max - clip_min)
            band_gamma = np.power(band_norm, gamma)
            img_out[:, :, i] = (band_gamma * 255).astype(np.uint8)
        return img_out

    def find_image_by_date(date):
        date_str = date.strftime('%Y%m%d')
        for fname in os.listdir(true_color_dir):
            if date_str in fname and fname.endswith('.tif'):
                return os.path.join(true_color_dir, fname)
        return None

    def read_true_color_image(path):
        try:
            ds = gdal.Open(path)
            if ds is None:
                return None
            # 读取3个波段，假设波段顺序是R,G,B
            bands = [ds.GetRasterBand(i).ReadAsArray().astype(np.float32) for i in [1, 2, 3]]
            img = np.stack(bands, axis=2)
            ds = None
            return img
        except Exception as e:
            print(f"读取影像失败: {e}")
            return None

    fig, ax = plt.subplots(figsize=(10, 8))

    def init():
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_title("异常点分布动画")
        ax.grid(True)

    def update(i):
        ax.clear()
        date = dates[i]
        daily_df = grouped.get_group(date)

        img_path = find_image_by_date(date)
        if img_path:
            img = read_true_color_image(img_path)
            if img is not None:
                height, width, _ = img.shape
                img_uint8 = fixed_clip_stretch(img, clip_min, clip_max, gamma)
                ax.imshow(img_uint8, extent=[0, width, height, 0])
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
        else:
            col_min, col_max = df['col'].min(), df['col'].max()
            row_min, row_max = df['row'].min(), df['row'].max()
            ax.set_xlim(col_min, col_max)
            ax.set_ylim(row_max, row_min)

        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_title(f"异常点分布 - {date}（共 {len(daily_df)} 个点）")
        ax.grid(True)
        ax.scatter(daily_df['col'], daily_df['row'], s=point_size, c='red', alpha=alpha)

    ani = FuncAnimation(fig, update, frames=len(dates), init_func=init,
                        repeat=False, interval=interval)

    ani.save(output_gif_path, writer='imagemagick', fps=fps)

    if show_jshtml:
        return HTML(ani.to_jshtml())
    else:
        plt.close(fig)
        print(f"✅ 动画已保存: {output_gif_path}")
