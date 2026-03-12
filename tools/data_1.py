import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================== 配置部分 ====================
# 1. 定义两个CSV文件路径
file1_path = r'D:\DS\Desktop\data_c\all_samples_33_1230.csv'
file2_path = r'D:\DS\Desktop\data_c\3_(1).csv'  # 修改为你的第二个文件

# 2. 定义要对比的列（可以来自不同文件的不同列）
compare_config = {
    'file1': {
        'file': file1_path,
        'column': 'feature_0',     # 文件1中的列名
        'color': 'blue',
        'label': '文件1 - 特征0'
    },
    'file2': {
        'file': file2_path,
        'column': '幅值 - 电压',     # 文件2中的列名
        'color': 'red',
        'label': '文件2 - 特征2'
    }
}

# 3. 其他参数
sample_index = 0       # 要对比的样本索引
n_points = 2048        # 要显示的数据点数
# =================================================

# 1. 读取数据
print("正在读取数据...")
df1 = pd.read_csv(compare_config['file1']['file'])
df2 = pd.read_csv(compare_config['file2']['file'])

print(f"文件1形状: {df1.shape}")
print(f"文件2形状: {df2.shape}")

# 2. 提取要对比的数据
# 提取文件1的数据
if 'sample_index' in df1.columns:
    data1 = df1[df1['sample_index'] == sample_index][compare_config['file1']['column']].values[:2048]
else:
    data1 = df1[compare_config['file1']['column']].values[:n_points]

# 提取文件2的数据  
if 'sample_index' in df2.columns:
    data2 = df2[df2['sample_index'] == sample_index][compare_config['file2']['column']].values[:2048]
else:
    data2 = df2[compare_config['file2']['column']].values[:n_points]

print(f"数据1长度: {len(data1)}")
print(f"数据2长度: {len(data2)}")

# 确保两个数据长度一致（取较小值）
min_len = min(len(data1), len(data2))
data1 = data1[:min_len]
data2 = data2[:min_len]

# 3. 创建对比图表
plt.figure(figsize=(16, 8))

# 绘制两条时间序列
x = np.arange(min_len)

plt.plot(x, data1, 
         color=compare_config['file1']['color'], 
         linewidth=1.5, 
         alpha=0.8,
         label=compare_config['file1']['label'])

plt.plot(x, data2, 
         color=compare_config['file2']['color'], 
         linewidth=1.5, 
         alpha=0.8,
         label=compare_config['file2']['label'])

# 4. 美化图表
plt.xlabel('时间点', fontsize=12)
plt.ylabel('特征值', fontsize=12)
plt.title(f'时间序列对比: {compare_config["file1"]["label"]} vs {compare_config["file2"]["label"]}', 
          fontsize=14, fontweight='bold')

plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3)

# 设置x轴刻度
if min_len > 100:
    step = min_len // 10
    plt.xticks(np.arange(0, min_len, step), fontsize=10)
plt.yticks(fontsize=10)

# 添加背景色区分
plt.axvspan(0, min_len, alpha=0.05, color='gray')

plt.tight_layout()

# 5. 保存图表
output_path = 'D:\DS\Desktop\data_c\comparison_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n对比图表已保存到: {output_path}")

plt.show()

# 6. 显示统计信息
print("\n" + "="*50)
print("统计信息对比:")
print("="*50)
print(f"{compare_config['file1']['label']}:")
print(f"  均值: {np.mean(data1):.4f}")
print(f"  标准差: {np.std(data1):.4f}")
print(f"  最小值: {np.min(data1):.4f}")
print(f"  最大值: {np.max(data1):.4f}")

print(f"\n{compare_config['file2']['label']}:")
print(f"  均值: {np.mean(data2):.4f}")
print(f"  标准差: {np.std(data2):.4f}")
print(f"  最小值: {np.min(data2):.4f}")
print(f"  最大值: {np.max(data2):.4f}")

# 计算相关性
correlation = np.corrcoef(data1, data2)[0, 1]
print(f"\n两个序列的相关系数: {correlation:.4f}")

# ================= 质量评估指标 =================
print("\n" + "="*50)
print("生成信号质量评估指标")
print("="*50)

# 1. MSE
mse = np.mean((data1 - data2) ** 2)
print(f"MSE (均方误差): {mse:.6f}")

# 2. MAE
mae = np.mean(np.abs(data1 - data2))
print(f"MAE (平均绝对误差): {mae:.6f}")

# 3. RMSE
rmse = np.sqrt(mse)
print(f"RMSE (均方根误差): {rmse:.6f}")

# 4. Pearson 相关（前面算过 correlation）
print(f"Pearson 相关系数: {correlation:.6f}")

# ========== 5. DTW 动态时间规整 ==========
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    
    dtw_distance, _ = fastdtw(data1, data2, dist=euclidean)
    print(f"DTW 距离: {dtw_distance:.4f}")
except ImportError:
    print("DTW 未安装，如需请执行: pip install fastdtw")

# ========== 6. 频域差异 ==========
# 傅里叶变换
fft1 = np.fft.fft(data1)
fft2 = np.fft.fft(data2)

# 计算频域能量差
freq_energy_diff = np.mean(np.abs(np.abs(fft1) - np.abs(fft2)))
print(f"频域能量差异: {freq_energy_diff:.6f}")

# 峰值频率对比
freq = np.fft.fftfreq(len(data1))
dom_freq1 = freq[np.argmax(np.abs(fft1))]
dom_freq2 = freq[np.argmax(np.abs(fft2))]
print(f"峰值主频(文件1): {dom_freq1:.4f}")
print(f"峰值主频(文件2): {dom_freq2:.4f}")