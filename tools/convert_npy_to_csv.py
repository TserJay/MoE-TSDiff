import numpy as np
import pandas as pd
import os

def convert_npy_to_csv(npy_path, output_dir, filename_prefix='generated_sample'):
    """
    将npy文件转换为多个csv文件
    
    Args:
        npy_path: .npy文件路径
        output_dir: 输出目录
        filename_prefix: 文件名前缀
    """
    # 加载数据
    data = np.load(npy_path)
    print(f"加载数据完成: {data.shape}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 每个样本有2048个时间步，3个特征
    num_samples = data.shape[0]
    seq_length = data.shape[1]
    
    # 创建完整的列索引 (0-5列，共6列)
    # 第2、4、6列对应索引1、3、5
    all_columns = list(range(6))  # [0, 1, 2, 3, 4, 5]
    
    for i in range(num_samples):
        # 创建一个6列的空DataFrame
        df = pd.DataFrame(0, index=range(seq_length), columns=all_columns)
        
        # 将三个特征分别放到第2、4、6列 (索引1, 3, 5)
        df.iloc[:, 1] = data[i, :, 0]  # 第2列 (索引1)
        df.iloc[:, 3] = data[i, :, 1]  # 第4列 (索引3)
        df.iloc[:, 5] = data[i, :, 2]  # 第6列 (索引5)
        
        # 保存为CSV文件
        filename = f"{filename_prefix}_{i+1}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False, header=True)
        
        print(f"保存 {filepath}")
    
    print(f"\n完成! 共保存 {num_samples} 个CSV文件到 {output_dir}")

if __name__ == '__main__':
    # 配置参数
    npy_file = 'Data/output/Leak/ddpm_fake_Leak_major0_sub0.npy'
    output_directory = 'Data/output/Leak/csv_samples_00'
    prefix = 'sample_major0_sub0'
    
    print(f"转换配置:")
    print(f"  输入文件: {npy_file}")
    print(f"  输出目录: {output_directory}")
    print(f"  文件前缀: {prefix}")
    print()
    
    convert_npy_to_csv(npy_file, output_directory, prefix)
