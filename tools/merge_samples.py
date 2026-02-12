import os
import pandas as pd

# 定义路径
input_dir = "/mnt/tj/projects/Diffusion-TS-main/Data/leak_signals/3/2"
output_file = "/mnt/tj/projects/Diffusion-TS-main/Data/leak_signals/3/2/merged_samples_50_2048.csv"

# 初始化合并数据框
merged_df = pd.DataFrame()

# 遍历50个样本文件
for i in range(1, 51):
    file_path = os.path.join(input_dir, f"({i}).csv")
    print(f"Processing file: {file_path}")
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 取前2048个时间点
    df = df.head(2048)
    
    # 添加sample_id列
    df['sample_id'] = i
    
    # 合并到总数据框
    merged_df = pd.concat([merged_df, df], ignore_index=True)

# 保存合并后的数据
merged_df.to_csv(output_file, index=False)
print(f"Merged data saved to: {output_file}")
print(f"Total samples: 50")
print(f"Total time points per sample: 2048")
print(f"Total time points in merged file: {len(merged_df)}")
