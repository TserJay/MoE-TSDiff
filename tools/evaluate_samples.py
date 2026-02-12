import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Generated Samples Quality')
    parser.add_argument('--generated_path', type=str, required=True,
                        help='Path to generated samples .npy file')
    parser.add_argument('--original_path', type=str, required=True,
                        help='Path to original training samples .npy file or directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
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
                samples.append(np.load(file_path))
        if samples:
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
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算统计指标
    original_stats = calculate_statistics(original)
    generated_stats = calculate_statistics(generated)
    
    # 计算分布距离
    kl_div = calculate_kl_divergence(original, generated)
    wasserstein_dist = calculate_wasserstein_distance(original, generated)
    
    # 计算相关性
    original_flat = original.reshape(-1, original.shape[-1])
    generated_flat = generated.reshape(-1, generated.shape[-1])
    correlation = np.corrcoef(original_flat.T, generated_flat.T)[:original.shape[-1], original.shape[-1]:]
    
    # 生成报告
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
        
        f.write('5. Quality Assessment\n')
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
        
        # 综合得分
        overall_score = (stat_score * 0.4 + kl_score * 0.2 + wasserstein_score * 0.2 + corr_score * 0.2)
        
        f.write(f'Statistical Similarity Score: {stat_score:.2f}/100\n')
        f.write(f'Distribution Similarity Score (KL): {kl_score:.2f}/100\n')
        f.write(f'Distribution Similarity Score (Wasserstein): {wasserstein_score:.2f}/100\n')
        f.write(f'Feature Correlation Score: {corr_score:.2f}/100\n')
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
    
    print(f'Evaluation report saved to: {report_path}')
    return overall_score

def main():
    args = parse_args()
    
    # 加载样本
    print('Loading samples...')
    original = load_samples(args.original_path)
    generated = load_samples(args.generated_path)
    
    print(f'Original samples shape: {original.shape}')
    print(f'Generated samples shape: {generated.shape}')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成评估报告
    print('Generating evaluation report...')
    overall_score = generate_evaluation_report(original, generated, args.output_dir)
    
    # 绘制统计指标对比
    print('Plotting statistical comparisons...')
    original_stats = calculate_statistics(original)
    generated_stats = calculate_statistics(generated)
    plot_statistics_comparison(original_stats, generated_stats, args.output_dir)
    
    # 绘制分布对比
    print('Plotting distribution comparisons...')
    plot_distribution_comparison(original, generated, args.output_dir, args.feature_idx)
    
    # 绘制样本对比
    print('Plotting sample comparisons...')
    plot_sample_comparison(original, generated, args.output_dir, args.n_samples, args.feature_idx)
    
    print(f'\nEvaluation completed!')
    print(f'Overall quality score: {overall_score:.2f}/100')
    print(f'Results saved to: {args.output_dir}')

if __name__ == '__main__':
    main()
