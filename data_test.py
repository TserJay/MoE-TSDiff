import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 加载数据
filled_data = np.load('/mnt/tj/ddpm_predict_Leak_33.npy')
print(f"数据形状: {filled_data.shape}")
print(f"数据类型: {filled_data.dtype}")

# 假设数据形状是 [样本数, 时间步数, 特征数]
# 例如: (n_samples, 1024, 3)

# 可视化（可选）
# plt.figure(figsize=(15, 5))
# for i in range(min(3, filled_data.shape[-1])):
#     plt.plot(filled_data[0, :, i], label=f'Feature {i+1}')
# plt.legend()
# plt.title('Time Series Data')
# plt.savefig('/mnt/tj/projects/MoE-TSDiff/Data/output/Leak_20251216_1553/samples/time_series_plot.png')
# plt.show()

# 保存所有样本到一个CSV文件
def save_all_samples_to_csv(data, output_path):
    """保存所有样本到CSV"""
    all_samples = []
    
    for sample_idx in range(data.shape[0]):
        for time_idx in range(data.shape[1]):
            row = {
                'sample_index': sample_idx,
                'time_step': time_idx,
            }
            # 添加每个特征的值
            for feature_idx in range(data.shape[2]):
                row[f'feature_{feature_idx}'] = data[sample_idx, time_idx, feature_idx]
            all_samples.append(row)
    
    df = pd.DataFrame(all_samples)
    df.to_csv(output_path, index=False)
    print(f"已保存到: {output_path}")
    print(f"总数据点: {len(df)}")
    return df

# 保存所有样本
output_csv = '/mnt/tj/projects/MoE-TSDiff/all_samples_33.csv'
df_all = save_all_samples_to_csv(filled_data, output_csv)

# 查看前几行
print("\nCSV文件前5行:")
print(df_all.head())