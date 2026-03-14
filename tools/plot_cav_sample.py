import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

"""
读取一个样本的全部通道数据并绘制可视化图表。

用法:
    # 绘制单个样本的所有通道
    python tools/plot_sample_channels.py --input_path ./csv_samples/D_1.csv --output_dir ./plots
    
    # 将所有通道绘制在同一个图上
    python tools/plot_sample_channels.py --input_path ./csv_samples/D_1.csv --output_dir ./plots --single_plot
"""

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def parse_args():
    parser = argparse.ArgumentParser(description='Plot all channels of a sample')
    parser.add_argument('--input_path', type=str, required=False,
                        default=r'Data/output/Leak/csv_samples_00/sample_major0_sub0_11.csv',
                        help='Path to sample CSV file')
    parser.add_argument('--output_dir', type=str, default='tools\plots',
                        help='Directory to save plots')
    parser.add_argument('--single_plot', action='store_true',default='True',
                        help='Plot all channels in a single figure')
    parser.add_argument('--figsize', type=int, nargs=2, default=[15, 10],
                        help='Figure size (width, height)')
    parser.add_argument('--channels', type=int, nargs='+', default=[1,3,5],
                        help='Column indices of channels (0-based). Default: [1, 3, 5] for columns 2, 4, 6')
    return parser.parse_args()

def load_sample(path, channels=None):
    """加载样本数据"""
    if os.path.isfile(path) and path.endswith('.csv'):
        # 读取CSV文件，假设没有表头
        df = pd.read_csv(path)
        
        # 如果指定了通道列，则只选择这些列
        if channels is not None:
            # 检查所有指定的列是否存在
            max_col = max(channels)
            if max_col >= df.shape[1]:
                raise ValueError(f"Channel index {max_col} out of range. File has only {df.shape[1]} columns.")
            return df.iloc[:, channels].values
        return df.values
    else:
        raise ValueError(f"Invalid path: {path}. Please provide a .csv file.")

def plot_single_channel(channel_data, channel_idx, ax):
    """绘制单个通道的数据"""
    ax.plot(channel_data)
    ax.set_title(f'Channel {channel_idx + 1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

def plot_all_channels_separate(sample_data, output_path, figsize=(15, 10)):
    """将所有通道绘制在单独的子图中"""
    n_channels = sample_data.shape[1]
    
    # 计算子图布局
    n_rows = (n_channels + 1) // 2  # 每行最多2个通道
    n_cols = min(n_channels, 2)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # 确保axes是一维数组
    
    # 绘制每个通道
    for i in range(n_channels):
        plot_single_channel(sample_data[:, i], i, axes[i])
    
    # 隐藏多余的子图
    for i in range(n_channels, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_all_channels_single(sample_data, output_path, figsize=(15, 10)):
    """将所有通道绘制在同一个图中"""
    n_channels = sample_data.shape[1]
    
    plt.figure(figsize=figsize)
    
    # 绘制每个通道
    for i in range(n_channels):
        plt.plot(sample_data[:, i], label=f'Channel {i + 1}')
    
    plt.title('All Channels')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_path):
        print(f"Error: Input file does not exist: {args.input_path}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory created: {args.output_dir}")
    
    # 加载样本数据
    print(f"Loading sample from: {args.input_path}")
    print(f"Selected channels (0-based): {args.channels}")
    sample_data = load_sample(args.input_path, channels=args.channels)
    print(f"Loaded sample shape: {sample_data.shape}")
    print(f"Number of channels: {sample_data.shape[1]}")
    print(f"Number of time steps: {sample_data.shape[0]}")
    
    # 生成输出文件名
    sample_name = os.path.splitext(os.path.basename(args.input_path))[0]
    
    # 绘制通道数据
    if args.single_plot:
        # 所有通道绘制在同一个图上
        output_path = os.path.join(args.output_dir, f"{sample_name}_all_channels.png")
        print(f"Plotting all channels in a single figure...")
        plot_all_channels_single(sample_data, output_path, figsize=tuple(args.figsize))
    else:
        # 每个通道绘制在单独的子图中
        output_path = os.path.join(args.output_dir, f"{sample_name}_channels.png")
        print(f"Plotting channels in separate subplots...")
        plot_all_channels_separate(sample_data, output_path, figsize=tuple(args.figsize))
    
    print(f"\nPlotting completed!")
    print(f"Plot saved to: {output_path}")

if __name__ == '__main__':
    main()