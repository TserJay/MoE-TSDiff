import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Generated Samples Quality')
    parser.add_argument('--generated_path', type=str, required=False,
                        default=r'D:\projects\Diffusion-TS-main\Data\output\Leak_2048_32_20260213\ddpm_fake_Leak.npy',
                        help='Path to generated samples .npy file')
    parser.add_argument('--original_path', type=str, required=False, 
                        default=r'D:\projects\Diffusion-TS-main\Data\output\Leak_2048_32_20260213\Leak_ground_truth_2048_train.npy',
                        help='Path to original training samples .npy file or directory')
    parser.add_argument('--output_dir', type=str, default='.\Data\output\Leak_2048_32_20260213\evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--feature_idx', type=int, default=0,
                        help='Feature index to visualize')
    return parser.parse_args()

def load_samples(path):
    """加载样本数据"""
    if os.path.isfile(path) and path.endswith('.npy'):
        return np.load(path)
    elif os.path.isdir(path):
        # 加载目录中的所有npy文件
        samples = []
        for file in os.listdir(path):
            if file.endswith('.npy'):
                file_path = os.path.join(path, file)
                sample = np.load(file_path)
                samples.append(sample)
        if samples:
            # 检查所有样本的形状是否一致
            first_shape = samples[0].shape
            for i, sample in enumerate(samples[1:], 1):
                if sample.shape != first_shape:
                    raise ValueError(f"All samples must have the same shape. Sample 0 has shape {first_shape}, sample {i} has shape {sample.shape}")
            return np.concatenate(samples, axis=0)
    raise ValueError(f"Invalid path: {path}. Please provide a .npy file or directory containing .npy files.")

def calculate_statistics(data):
    """计算统计指标"""
    stats_dict = {
        'mean': np.mean(data, axis=(0, 1)),
        'std': np.std(data, axis=(0, 1)),
        'max': np.max(data, axis=(0, 1)),
        'min': np.min(data, axis=(0, 1)),
        'median': np.median(data, axis=(0, 1)),
        'q25': np.percentile(data, 25, axis=(0, 1)),
        'q75': np.percentile(data, 75, axis=(0, 1)),
        'skewness': stats.skew(data.reshape(-1, data.shape[-1]), axis=0),
        'kurtosis': stats.kurtosis(data.reshape(-1, data.shape[-1]), axis=0)
    }
    return stats_dict

def calculate_kl_divergence(p, q, bins=100):
    """计算KL散度"""
    p_flat = p.reshape(-1)
    q_flat = q.reshape(-1)
    
    min_val = min(np.min(p_flat), np.min(q_flat))
    max_val = max(np.max(p_flat), np.max(q_flat))
    
    p_hist, _ = np.histogram(p_flat, bins=bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_flat, bins=bins, range=(min_val, max_val), density=True)
    
    # 避免除零错误
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    
    return stats.entropy(p_hist, q_hist)

def calculate_wasserstein_distance(p, q):
    """计算Wasserstein距离"""
    p_flat = p.reshape(-1)
    q_flat = q.reshape(-1)
    return stats.wasserstein_distance(p_flat, q_flat)

def calculate_discriminative_score(original, generated):
    """计算判别性得分"""
    print(' Calculating discriminative score...')
    
    # 准备数据
    original_flat = original.reshape(len(original), -1)
    generated_flat = generated.reshape(len(generated), -1)
    
    # 限制样本数量以避免计算过载
    n_samples = min(1000, len(original_flat), len(generated_flat))
    original_subset = original_flat[:n_samples]
    generated_subset = generated_flat[:n_samples]
    
    # 创建标签（0: 原始数据, 1: 生成数据）
    original_labels = np.zeros(len(original_subset))
    generated_labels = np.ones(len(generated_subset))
    
    # 合并数据
    X = np.vstack([original_subset, generated_subset])
    y = np.hstack([original_labels, generated_labels])
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 训练判别器
    discriminator = LogisticRegression(max_iter=1000)
    discriminator.fit(X_train, y_train)
    
    # 评估判别器
    y_pred = discriminator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 计算判别性得分
    discriminative_score = 1 - accuracy
    
    print(f' Discriminative score calculated: {discriminative_score}')
    return discriminative_score

def calculate_predictive_score(original, generated):
    """计算预测性得分"""
    print(' Calculating predictive score...')
    
    # 对于时间序列，我们使用简单的自回归预测
    # 这里使用最后一个时间步预测第一个时间步作为示例
    def prepare_autoregression_data(data):
        X = data[:, :-1, :].reshape(len(data), -1)  # 除了最后一个时间步的所有数据
        y = data[:, -1, 0]  # 最后一个时间步的第一个特征
        return X, y
    
    # 准备数据
    X_original, y_original = prepare_autoregression_data(original)
    X_generated, y_generated = prepare_autoregression_data(generated)
    
    # 限制样本数量以避免计算过载
    n_samples = min(1000, len(X_original), len(X_generated))
    X_generated_subset = X_generated[:n_samples]
    y_generated_subset = y_generated[:n_samples]
    X_original_subset = X_original[:n_samples]
    y_original_subset = y_original[:n_samples]
    
    # 数据标准化
    scaler = StandardScaler()
    X_generated_scaled = scaler.fit_transform(X_generated_subset)
    X_original_scaled = scaler.transform(X_original_subset)
    
    # 训练预测模型
    predictor = LogisticRegression(max_iter=1000)
    predictor.fit(X_generated_scaled, y_generated_subset > 0)  # 二分类预测
    
    # 在原始数据上评估
    y_pred = predictor.predict(X_original_scaled)
    accuracy = accuracy_score(y_original_subset > 0, y_pred)
    
    print(f' Predictive score calculated: {accuracy}')
    return accuracy

def calculate_correlational_score(original, generated):
    """计算相关性得分"""
    print(' Calculating correlational score...')
    
    # 计算原始数据的特征相关矩阵
    original_flat = original.reshape(-1, original.shape[-1])
    original_corr = np.corrcoef(original_flat.T)
    
    # 计算生成数据的特征相关矩阵
    generated_flat = generated.reshape(-1, generated.shape[-1])
    generated_corr = np.corrcoef(generated_flat.T)
    
    # 计算两个相关矩阵之间的相似度（使用Frobenius范数）
    corr_diff = np.linalg.norm(original_corr - generated_corr, 'fro')
    max_diff = np.linalg.norm(np.ones_like(original_corr), 'fro')
    correlational_score = 1 - (corr_diff / max_diff)
    
    print(f' Correlational score calculated: {correlational_score}')
    return correlational_score

def calculate_context_fid_score(original, generated):
    """计算上下文FID得分"""
    print(' Calculating context FID score...')
    
    # 对于时间序列，我们使用简单的特征提取方法
    # 计算每个样本的统计特征作为表示
    def extract_features(data):
        features = []
        for sample in data:
            sample_features = []
            for i in range(sample.shape[-1]):
                feature = sample[:, i]
                sample_features.extend([
                    np.mean(feature),
                    np.std(feature),
                    np.max(feature),
                    np.min(feature),
                    np.median(feature),
                    stats.skew(feature),
                    stats.kurtosis(feature)
                ])
            features.append(sample_features)
        return np.array(features)
    
    # 提取特征
    original_features = extract_features(original)
    generated_features = extract_features(generated)
    
    # 计算均值和协方差
    mu_original = np.mean(original_features, axis=0)
    sigma_original = np.cov(original_features.T)
    
    mu_generated = np.mean(generated_features, axis=0)
    sigma_generated = np.cov(generated_features.T)
    
    # 计算FID得分
    try:
        # 计算FID
        diff = mu_original - mu_generated
        covmean = np.sqrt(np.linalg.det(sigma_original) * np.linalg.det(sigma_generated))
        if covmean < 1e-10:
            covmean = 1e-10
        fid = np.dot(diff, diff) + np.trace(sigma_original + sigma_generated - 2 * covmean)
        # 转换为0-1的得分
        context_fid_score = max(0, 1 - fid / 100)
    except Exception as e:
        print(f' Error calculating FID: {e}')
        context_fid_score = 0.5  # 默认值
    
    print(f' Context FID score calculated: {context_fid_score}')
    return context_fid_score

def plot_statistics_comparison(original_stats, generated_stats, output_dir):
    """绘制统计指标对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    features = range(len(original_stats['mean']))
    stats_keys = ['mean', 'std', 'max', 'min', 'median', 'skewness', 'kurtosis']
    
    for stat_key in stats_keys:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(features))
        width = 0.35
        
        plt.bar(x - width/2, original_stats[stat_key], width, label='Original')
        plt.bar(x + width/2, generated_stats[stat_key], width, label='Generated')
        
        plt.xlabel('Feature Index')
        plt.ylabel(stat_key.capitalize())
        plt.title(f'{stat_key.capitalize()} Comparison')
        plt.xticks(x, features)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{stat_key}_comparison.png'))
        plt.close()

def plot_distribution_comparison(original, generated, output_dir, feature_idx=0):
    """绘制分布对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取指定特征
    original_feature = original[:, :, feature_idx].reshape(-1)
    generated_feature = generated[:, :, feature_idx].reshape(-1)
    
    plt.figure(figsize=(12, 6))
    
    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(original_feature, bins=50, alpha=0.5, label='Original', density=True)
    plt.hist(generated_feature, bins=50, alpha=0.5, label='Generated', density=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram Comparison')
    plt.legend()
    
    # 核密度估计
    plt.subplot(1, 2, 2)
    sns.kdeplot(original_feature, label='Original')
    sns.kdeplot(generated_feature, label='Generated')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('KDE Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_comparison_feature_{feature_idx}.png'))
    plt.close()

def plot_sample_comparison(original, generated, output_dir, n_samples=5, feature_idx=0):
    """绘制样本对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择样本
    original_idx = np.random.choice(len(original), n_samples, replace=False)
    generated_idx = np.random.choice(len(generated), n_samples, replace=False)
    
    for i, (orig_idx, gen_idx) in enumerate(zip(original_idx, generated_idx)):
        plt.figure(figsize=(15, 6))
        
        # 原始样本
        plt.subplot(1, 2, 1)
        plt.plot(original[orig_idx, :, feature_idx])
        plt.title(f'Original Sample {orig_idx}')
        plt.xlabel('Time Step')
        plt.ylabel(f'Feature {feature_idx}')
        
        # 生成样本
        plt.subplot(1, 2, 2)
        plt.plot(generated[gen_idx, :, feature_idx])
        plt.title(f'Generated Sample {gen_idx}')
        plt.xlabel('Time Step')
        plt.ylabel(f'Feature {feature_idx}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_comparison_{i}.png'))
        plt.close()

def generate_evaluation_report(original, generated, output_dir):
    """生成综合评估报告"""
    print(f'Starting generate_evaluation_report')
    print(f'Output directory: {output_dir}')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory created')
    
    # 计算统计指标
    print(f'Calculating statistics...')
    original_stats = calculate_statistics(original)
    generated_stats = calculate_statistics(generated)
    print(f'Statistics calculated successfully')
    
    # 计算分布距离
    print(f'Calculating distribution metrics...')
    kl_div = calculate_kl_divergence(original, generated)
    wasserstein_dist = calculate_wasserstein_distance(original, generated)
    print(f'Distribution metrics calculated successfully')
    print(f'KL Divergence: {kl_div}')
    print(f'Wasserstein Distance: {wasserstein_dist}')
    
    # 计算相关性 - 使用特征级别的统计指标相关性
    print(f'Calculating correlation...')
    original_feature_stats = []
    generated_feature_stats = []
    
    for i in range(original.shape[-1]):
        # 提取单个特征
        original_feature = original[:, :, i]
        generated_feature = generated[:, :, i]
        
        # 计算每个特征的统计指标
        orig_stats = {
            'mean': np.mean(original_feature),
            'std': np.std(original_feature),
            'max': np.max(original_feature),
            'min': np.min(original_feature),
            'median': np.median(original_feature)
        }
        
        gen_stats = {
            'mean': np.mean(generated_feature),
            'std': np.std(generated_feature),
            'max': np.max(generated_feature),
            'min': np.min(generated_feature),
            'median': np.median(generated_feature)
        }
        
        original_feature_stats.append(list(orig_stats.values()))
        generated_feature_stats.append(list(gen_stats.values()))
    
    # 计算特征统计指标之间的相关性
    correlation = np.corrcoef(original_feature_stats, generated_feature_stats)[:original.shape[-1], original.shape[-1]:]
    print(f'Correlation calculated successfully')
    print(f'Correlation matrix: {correlation}')
    
    # 计算新指标
    print(f'Calculating new metrics...')
    discriminative_score = calculate_discriminative_score(original, generated)
    predictive_score = calculate_predictive_score(original, generated)
    correlational_score = calculate_correlational_score(original, generated)
    context_fid_score = calculate_context_fid_score(original, generated)
    print(f'New metrics calculated successfully')
    
    # 生成报告
    print(f'Generating report...')
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('=== Sample Quality Evaluation Report ===\n\n')
        
        f.write('1. Data Overview\n')
        f.write(f'Original samples shape: {original.shape}\n')
        f.write(f'Generated samples shape: {generated.shape}\n')
        f.write(f'Number of features: {original.shape[-1]}\n')
        f.write(f'Number of time steps: {original.shape[1]}\n\n')
        
        f.write('2. Statistical Metrics\n')
        f.write('-' * 50 + '\n')
        for key in original_stats:
            f.write(f'{key.upper()}:\n')
            f.write(f'  Original: {original_stats[key]}\n')
            f.write(f'  Generated: {generated_stats[key]}\n')
            f.write(f'  Difference: {np.abs(original_stats[key] - generated_stats[key])}\n')
            f.write('\n')
        
        f.write('3. Distribution Metrics\n')
        f.write('-' * 50 + '\n')
        f.write(f'KL Divergence: {kl_div}\n')
        f.write(f'Wasserstein Distance: {wasserstein_dist}\n\n')
        
        f.write('4. Correlation Matrix\n')
        f.write('-' * 50 + '\n')
        f.write('Correlation between original and generated features:\n')
        f.write(str(correlation) + '\n\n')
        
        # 添加新指标报告
        f.write('5. Advanced Quality Metrics\n')
        f.write('-' * 50 + '\n')
        f.write(f'Discriminative Score: {discriminative_score:.4f} (closer to 0.5 is better)\n')
        f.write(f'Predictive Score: {predictive_score:.4f} (higher is better)\n')
        f.write(f'Correlational Score: {correlational_score:.4f} (higher is better)\n')
        f.write(f'Context FID Score: {context_fid_score:.4f} (higher is better)\n\n')
        
        f.write('6. Quality Assessment\n')
        f.write('-' * 50 + '\n')
        
        # 计算综合得分
        # 基于统计指标差异
        stat_diff = 0
        for key in ['mean', 'std', 'max', 'min', 'median']:
            stat_diff += np.mean(np.abs(original_stats[key] - generated_stats[key]))
        stat_score = max(0, 100 - stat_diff * 10)
        
        # 基于分布距离
        kl_score = max(0, 100 - kl_div * 10)
        wasserstein_score = max(0, 100 - wasserstein_dist * 100)
        
        # 基于相关性
        corr_score = np.mean(np.diag(correlation)) * 100
        
        # 基于新指标
        discriminative_metric = max(0, 100 - abs(discriminative_score - 0.5) * 200)  # 0.5为最佳
        predictive_metric = predictive_score * 100
        correlational_metric = correlational_score * 100
        context_fid_metric = context_fid_score * 100
        
        # 综合得分 - 调整权重以包含新指标
        overall_score = (
            stat_score * 0.2 + 
            kl_score * 0.1 + 
            wasserstein_score * 0.1 + 
            corr_score * 0.1 +
            discriminative_metric * 0.15 +
            predictive_metric * 0.15 +
            correlational_metric * 0.1 +
            context_fid_metric * 0.1
        )
        
        f.write(f'Statistical Similarity Score: {stat_score:.2f}/100\n')
        f.write(f'Distribution Similarity Score (KL): {kl_score:.2f}/100\n')
        f.write(f'Distribution Similarity Score (Wasserstein): {wasserstein_score:.2f}/100\n')
        f.write(f'Feature Correlation Score: {corr_score:.2f}/100\n')
        f.write(f'Discriminative Metric: {discriminative_metric:.2f}/100\n')
        f.write(f'Predictive Metric: {predictive_metric:.2f}/100\n')
        f.write(f'Correlational Metric: {correlational_metric:.2f}/100\n')
        f.write(f'Context FID Metric: {context_fid_metric:.2f}/100\n')
        f.write(f'Overall Quality Score: {overall_score:.2f}/100\n\n')
        
        # 评估结论
        if overall_score >= 80:
            f.write('Conclusion: Excellent! Generated samples closely match original samples.\n')
        elif overall_score >= 60:
            f.write('Conclusion: Good. Generated samples have reasonable similarity to original samples.\n')
        elif overall_score >= 40:
            f.write('Conclusion: Fair. Generated samples show some similarity but have noticeable differences.\n')
        else:
            f.write('Conclusion: Poor. Generated samples significantly differ from original samples.\n')
    
    print(f' Report generated successfully: {report_path}')
    print(f' Overall score: {overall_score}')
    
    print(f'Evaluation report saved to: {report_path}')
    return overall_score

def main():
    args = parse_args()
    
    print(f'Command line arguments:')
    print(f'generated_path: {args.generated_path}')
    print(f'original_path: {args.original_path}')
    print(f'output_dir: {args.output_dir}')
    
    # 检查输入文件是否存在
    print('Checking input files...')
    if not os.path.exists(args.original_path):
        print(f'Error: Original path does not exist: {args.original_path}')
        return
    if not os.path.exists(args.generated_path):
        print(f'Error: Generated path does not exist: {args.generated_path}')
        return
    print('Input files exist.')
    
    # 加载样本
    print('Loading samples...')
    original = load_samples(args.original_path)
    generated = load_samples(args.generated_path)
    
    print(f'Original samples shape: {original.shape}')
    print(f'Generated samples shape: {generated.shape}')
    
    # 创建输出目录
    print(f'Creating output directory: {args.output_dir}')
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'Output directory created successfully: {args.output_dir}')
    except Exception as e:
        print(f'Error creating output directory: {e}')
        return
    
    # 生成评估报告
    print('Generating evaluation report...')
    try:
        overall_score = generate_evaluation_report(original, generated, args.output_dir)
        print(f'Evaluation report generated successfully')
    except Exception as e:
        print(f'Error generating evaluation report: {e}')
        import traceback
        traceback.print_exc()
        return
    
    # # 绘制统计指标对比
    # print('Plotting statistical comparisons...')
    # try:
    #     original_stats = calculate_statistics(original)
    #     generated_stats = calculate_statistics(generated)
    #     plot_statistics_comparison(original_stats, generated_stats, args.output_dir)
    #     print('Statistical comparisons plotted successfully')
    # except Exception as e:
    #     print(f'Error plotting statistical comparisons: {e}')
    #     import traceback
    #     traceback.print_exc()
    
    # # 绘制分布对比
    # print('Plotting distribution comparisons...')
    # try:
    #     plot_distribution_comparison(original, generated, args.output_dir, args.feature_idx)
    #     print('Distribution comparisons plotted successfully')
    # except Exception as e:
    #     print(f'Error plotting distribution comparisons: {e}')
    #     import traceback
    #     traceback.print_exc()
    
    # # 绘制样本对比
    # print('Plotting sample comparisons...')
    # try:
    #     plot_sample_comparison(original, generated, args.output_dir, args.n_samples, args.feature_idx)
    #     print('Sample comparisons plotted successfully')
    # except Exception as e:
    #     print(f'Error plotting sample comparisons: {e}')
    #     import traceback
    #     traceback.print_exc()
    
    print(f'\nEvaluation completed!')
    print(f'Overall quality score: {overall_score:.2f}/100')
    print(f'Results saved to: {args.output_dir}')

if __name__ == '__main__':
    main()
