import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Generated Samples Quality')
    parser.add_argument('--generated_path', type=str, required=False,
                        default=r'Data\output\Leak\ddpm_fake_Leak_major3_sub2.npy',
                        help='Path to generated samples .npy file')
    parser.add_argument('--original_path', type=str, required=False, 
                        default=r'Data\leak_signals\3\12',   
                        help='Path to original training samples directory')
    parser.add_argument('--output_dir', type=str, default='.\Data\output\Leak\evaluation_results_32-311',
                        help='Directory to save evaluation results')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--feature_idx', type=int, default=0,
                        help='Feature index to visualize')
    return parser.parse_args()

def load_samples(path):
    """加载样本数据"""
    if os.path.isdir(path):
        if 'leak_signals' in path.lower():
            return load_samples_from_csv(path)
        else:
            samples = []
            for file in os.listdir(path):
                if file.endswith('.npy'):
                    samples.append(np.load(os.path.join(path, file)))
            if samples:
                return np.concatenate(samples, axis=0)
    elif os.path.isfile(path) and path.endswith('.npy'):
        return np.load(path)
    raise ValueError(f"Invalid path: {path}")

def load_samples_from_csv(data_root, use_cols=None, window=2048):
    """从CSV文件读取数据"""
    if use_cols is None:
        use_cols = [1, 3, 5]
    
    all_data = []
    for csv_file in sorted(os.listdir(data_root)):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(data_root, csv_file)
            try:
                df = pd.read_csv(csv_path, header=0)
                data = df.iloc[:, use_cols].values
                if len(data) >= window:
                    all_data.append(data[:window, :])
            except Exception as e:
                print(f"Warning: Failed to read {csv_path}: {e}")
    
    if not all_data:
        raise ValueError(f"No valid data found in {data_root}")
    
    samples = np.array(all_data, dtype=np.float32)
    print(f"Loaded {len(samples)} samples from CSV files")
    print(f"Samples shape: {samples.shape}")
    return samples

def calculate_statistics(data):
    """计算统计指标"""
    return {
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

def calculate_kl_divergence(p, q, bins=100):
    """计算KL散度"""
    p_flat, q_flat = p.reshape(-1), q.reshape(-1)
    min_val, max_val = min(np.min(p_flat), np.min(q_flat)), max(np.max(p_flat), np.max(q_flat))
    p_hist, _ = np.histogram(p_flat, bins=bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_flat, bins=bins, range=(min_val, max_val), density=True)
    return stats.entropy(p_hist + 1e-10, q_hist + 1e-10)

def calculate_wasserstein_distance(p, q):
    """计算Wasserstein距离"""
    return stats.wasserstein_distance(p.reshape(-1), q.reshape(-1))

def calculate_discriminative_score(original, generated):
    """计算判别性得分"""
    print(' Calculating discriminative score...')
    n_samples = min(1000, len(original), len(generated))
    X = np.vstack([original[:n_samples].reshape(n_samples, -1), 
                   generated[:n_samples].reshape(n_samples, -1)])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    discriminator = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    return 1 - accuracy_score(y_test, discriminator.predict(X_test))

def calculate_predictive_score(original, generated):
    """计算预测性得分"""
    print(' Calculating predictive score...')
    n_samples = min(1000, len(original), len(generated))
    X_orig, y_orig = original[:n_samples, :-1, :].reshape(n_samples, -1), original[:n_samples, -1, 0]
    X_gen, y_gen = generated[:n_samples, :-1, :].reshape(n_samples, -1), generated[:n_samples, -1, 0]
    scaler = StandardScaler()
    X_gen_scaled = scaler.fit_transform(X_gen)
    X_orig_scaled = scaler.transform(X_orig)
    predictor = LogisticRegression(max_iter=1000).fit(X_gen_scaled, y_gen > 0)
    return accuracy_score(y_orig > 0, predictor.predict(X_orig_scaled))

def calculate_correlational_score(original, generated):
    """计算相关性得分"""
    print(' Calculating correlational score...')
    orig_corr = np.corrcoef(original.reshape(-1, original.shape[-1]).T)
    gen_corr = np.corrcoef(generated.reshape(-1, generated.shape[-1]).T)
    return 1 - np.linalg.norm(orig_corr - gen_corr, 'fro') / np.linalg.norm(np.ones_like(orig_corr), 'fro')

def calculate_context_fid_score(original, generated):
    """计算上下文FID得分"""
    print(' Calculating context FID score...')
    def extract_features(data):
        features = []
        for sample in data:
            for i in range(sample.shape[-1]):
                feature = sample[:, i]
                features.extend([np.mean(feature), np.std(feature), np.max(feature), 
                               np.min(feature), np.median(feature), 
                               stats.skew(feature), stats.kurtosis(feature)])
        return np.array(features).reshape(len(data), -1)
    
    orig_feat, gen_feat = extract_features(original), extract_features(generated)
    diff = np.mean(orig_feat, axis=0) - np.mean(gen_feat, axis=0)
    try:
        fid = np.dot(diff, diff) + np.trace(np.cov(orig_feat.T) + np.cov(gen_feat.T))
        return max(0, 1 - fid / 100)
    except:
        return 0.5

def plot_statistics_comparison(original_stats, generated_stats, output_dir):
    """绘制统计指标对比图"""
    os.makedirs(output_dir, exist_ok=True)
    for stat_key in ['mean', 'std', 'max', 'min', 'median']:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(original_stats[stat_key]))
        plt.bar(x - 0.15, original_stats[stat_key], 0.3, label='Original')
        plt.bar(x + 0.15, generated_stats[stat_key], 0.3, label='Generated')
        plt.xlabel('Feature Index')
        plt.ylabel(stat_key.capitalize())
        plt.title(f'{stat_key.capitalize()} Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{stat_key}_comparison.png'))
        plt.close()

def plot_distribution_comparison(original, generated, output_dir, feature_idx=0):
    """绘制分布对比图"""
    os.makedirs(output_dir, exist_ok=True)
    orig_feat = original[:, :, feature_idx].reshape(-1)
    gen_feat = generated[:, :, feature_idx].reshape(-1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(orig_feat, bins=50, alpha=0.5, label='Original', density=True)
    plt.hist(gen_feat, bins=50, alpha=0.5, label='Generated', density=True)
    plt.xlabel('Value'); plt.ylabel('Density'); plt.title('Histogram')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(orig_feat, label='Original'); sns.kdeplot(gen_feat, label='Generated')
    plt.xlabel('Value'); plt.ylabel('Density'); plt.title('KDE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'))
    plt.close()

def plot_sample_comparison(original, generated, output_dir, n_samples=5, feature_idx=0):
    """绘制样本对比图"""
    os.makedirs(output_dir, exist_ok=True)
    orig_idx = np.random.choice(len(original), n_samples, replace=False)
    gen_idx = np.random.choice(len(generated), n_samples, replace=False)
    
    for i, (oi, gi) in enumerate(zip(orig_idx, gen_idx)):
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(original[oi, :, feature_idx])
        plt.title(f'Original Sample {i}'); plt.xlabel('Time'); plt.ylabel(f'Feature {feature_idx}')
        plt.subplot(1, 2, 2)
        plt.plot(generated[gi, :, feature_idx])
        plt.title(f'Generated Sample {i}'); plt.xlabel('Time'); plt.ylabel(f'Feature {feature_idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_comparison_{i}.png'))
        plt.close()

def generate_evaluation_report(original, generated, output_dir):
    """生成综合评估报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算指标
    orig_stats = calculate_statistics(original)
    gen_stats = calculate_statistics(generated)
    kl_div = calculate_kl_divergence(original, generated)
    wasserstein_dist = calculate_wasserstein_distance(original, generated)
    
    # 计算特征统计相关性
    orig_stats_list = [[np.mean(original[:,:,i]), np.std(original[:,:,i]), 
                       np.max(original[:,:,i]), np.min(original[:,:,i]), 
                       np.median(original[:,:,i])] for i in range(original.shape[-1])]
    gen_stats_list = [[np.mean(generated[:,:,i]), np.std(generated[:,:,i]), 
                      np.max(generated[:,:,i]), np.min(generated[:,:,i]), 
                      np.median(generated[:,:,i])] for i in range(generated.shape[-1])]
    correlation = np.corrcoef(orig_stats_list, gen_stats_list)[:original.shape[-1], original.shape[-1]:]
    
    # 计算新指标
    disc_score = calculate_discriminative_score(original, generated)
    pred_score = calculate_predictive_score(original, generated)
    corr_score = calculate_correlational_score(original, generated)
    fid_score = calculate_context_fid_score(original, generated)
    
    # 生成报告
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('=== Sample Quality Evaluation Report ===\n\n')
        f.write(f'Original shape: {original.shape}\n')
        f.write(f'Generated shape: {generated.shape}\n\n')
        
        f.write('2. Statistical Metrics\n')
        for key in orig_stats:
            f.write(f'{key.upper()}:\n  Original: {orig_stats[key]}\n  Generated: {gen_stats[key]}\n')
        
        f.write(f'\n3. Distribution Metrics\n')
        f.write(f'KL Divergence: {kl_div:.4f}\n')
        f.write(f'Wasserstein Distance: {wasserstein_dist:.4f}\n')
        
        f.write(f'\n4. Correlation Matrix\n')
        f.write(f'{correlation}\n')
        
        f.write(f'\n5. Advanced Metrics\n')
        f.write(f'Discriminative Score: {disc_score:.4f}\n')
        f.write(f'Predictive Score: {pred_score:.4f}\n')
        f.write(f'Correlational Score: {corr_score:.4f}\n')
        f.write(f'Context FID Score: {fid_score:.4f}\n')
        
        # 综合得分
        stat_score = max(0, 100 - (kl_div * 10 + wasserstein_dist * 10))
        disc_metric = max(0, 100 - abs(disc_score - 0.5) * 200)
        pred_metric = pred_score * 100
        corr_metric = (np.mean(np.diag(correlation)) * 100) if correlation.size > 0 else 0
        fid_metric = fid_score * 100
        
        overall = (stat_score * 0.2 + disc_metric * 0.15 + pred_metric * 0.15 + 
                  corr_metric * 0.1 + fid_metric * 0.1)
        
        f.write(f'\n6. Quality Scores\n')
        f.write(f'Statistical Score: {stat_score:.2f}/100\n')
        f.write(f'Discriminative Metric: {disc_metric:.2f}/100\n')
        f.write(f'Predictive Metric: {pred_metric:.2f}/100\n')
        f.write(f'Correlational Metric: {corr_metric:.2f}/100\n')
        f.write(f'Context FID Metric: {fid_metric:.2f}/100\n')
        f.write(f'\nOverall Quality Score: {overall:.2f}/100\n')
        
        if overall >= 80:
            f.write('\nConclusion: Excellent!\n')
        elif overall >= 60:
            f.write('\nConclusion: Good.\n')
        elif overall >= 40:
            f.write('\nConclusion: Fair.\n')
        else:
            f.write('\nConclusion: Poor.\n')
    
    print(f'Report saved to: {report_path}')
    return overall

def main():
    args = parse_args()
    
    print(f'Command line arguments:')
    print(f'generated_path: {args.generated_path}')
    print(f'original_path: {args.original_path}')
    print(f'output_dir: {args.output_dir}')
    
    if not os.path.exists(args.generated_path):
        print(f'Error: Generated path does not exist: {args.generated_path}')
        return
    
    print('Loading samples...')
    generated = load_samples(args.generated_path)
    print(f'Generated samples shape: {generated.shape}')
    
    original = load_samples(args.original_path)
    print(f'Original samples shape: {original.shape}')
    
    print(f'Creating output directory: {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print('Generating evaluation report...')
    overall_score = generate_evaluation_report(original, generated, args.output_dir)
    
    print('Plotting comparisons...')
    plot_statistics_comparison(calculate_statistics(original), calculate_statistics(generated), args.output_dir)
    plot_distribution_comparison(original, generated, args.output_dir, args.feature_idx)
    plot_sample_comparison(original, generated, args.output_dir, args.n_samples, args.feature_idx)
    
    print(f'\nEvaluation completed!')
    print(f'Overall quality score: {overall_score:.2f}/100')
    print(f'Results saved to: {args.output_dir}')

if __name__ == '__main__':
    main()
