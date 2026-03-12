# 样本质量评估工具

## 功能说明

这个工具用于评估生成的时间序列样本与原始训练样本的质量对比，包括：

- 统计指标对比（均值、标准差、最大值、最小值等）
- 分布对比（直方图、核密度估计）
- 分布距离计算（KL散度、Wasserstein距离）
- 特征相关性分析
- 样本可视化对比
- 综合质量评估报告

## 安装依赖

```bash
pip install numpy matplotlib seaborn scipy
```

## 使用方法

### 基本用法

```bash
python evaluate_samples.py \
    --generated_path /path/to/generated_samples.npy \
    --original_path /path/to/original_samples.npy \
    --output_dir ./evaluation_results
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--generated_path` | 生成样本的路径（.npy文件或包含.npy文件的目录） | 必填 |
| `--original_path` | 原始样本的路径（.npy文件或包含.npy文件的目录） | 必填 |
| `--output_dir` | 评估结果保存目录 | `./evaluation_results` |
| `--n_samples` | 要可视化的样本数量 | `5` |
| `--feature_idx` | 要可视化的特征索引 | `0` |

## 示例

### 评估生成样本

```bash
python evaluate_samples.py \
    --generated_path ../Data/output/Leak/ddpm_fake_Leak.npy \
    --original_path ../Data/leak_signals/3/2/merged_samples_50_2048.npy \
    --output_dir ../evaluation_results/Leak
```

## 输出结果

运行后会在指定的输出目录生成以下文件：

### 1. 评估报告
- `evaluation_report.txt` - 详细的评估报告，包含统计指标、分布距离、相关性和综合质量得分

### 2. 统计指标对比图
- `mean_comparison.png`
- `std_comparison.png`
- `max_comparison.png`
- `min_comparison.png`
- `median_comparison.png`
- `skewness_comparison.png`
- `kurtosis_comparison.png`

### 3. 分布对比图
- `distribution_comparison_feature_0.png` - 特征0的分布对比

### 4. 样本对比图
- `sample_comparison_0.png`
- `sample_comparison_1.png`
- `sample_comparison_2.png`
- `sample_comparison_3.png`
- `sample_comparison_4.png`

## 质量评估标准

综合质量得分（0-100）：
- **80-100**：优秀 - 生成样本与原始样本高度相似
- **60-79**：良好 - 生成样本与原始样本有合理的相似性
- **40-59**：一般 - 生成样本显示出一定的相似性，但有明显差异
- **0-39**：较差 - 生成样本与原始样本有显著差异

## 注意事项

1. 确保原始样本和生成样本的形状一致（样本数可以不同，但时间步数和特征数必须相同）
2. 对于大型数据集，评估可能需要较长时间和较多内存
3. 可视化功能需要Matplotlib库支持
4. 分布距离计算可能对离群值敏感，建议在评估前对数据进行适当处理
