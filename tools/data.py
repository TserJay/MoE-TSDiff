import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
csv_path = 'all_samples_1219.csv'
df = pd.read_csv(csv_path)

# 2. 提取样本0的前2048行
# sample_0 = df[df['sample_index'] == 0].head(2048)
sample_0 = df.head(2048)

# 3. 创建图表
plt.figure(figsize=(18, 8))

# 设置时间步为x轴（如果time_step列存在）
if 'time_step' in sample_0.columns:
    x = sample_0['time_step'].values
else:
    x = np.arange(len(sample_0))

# 用三种不同颜色画出三个特征
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
feature_names = ['feature_0', 'feature_1', 'feature_2']
labels = ['特征 0', '特征 1', '特征 2']

for i, (feature, color, label) in enumerate(zip(feature_names, colors, labels)):
    plt.plot(x, sample_0[feature].values, 
             color=color, 
             linewidth=1.2, 
             alpha=0.8,
             label=label)

# 4. 设置图表样式
plt.xlabel('时间点', fontsize=12)
plt.ylabel('特征值', fontsize=12)
plt.title('样本 0 - 三个特征的时间序列 (2048个时间点)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3)

# 设置x轴刻度
plt.xticks(np.arange(0, 2048, 256), fontsize=10)
plt.yticks(fontsize=10)

# 调整布局
plt.tight_layout()

# 5. 保存图像
output_path = 'D:\DS\Desktop\data_c\sample_0_features_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已保存到: {output_path}")

# 显示图表
plt.show()