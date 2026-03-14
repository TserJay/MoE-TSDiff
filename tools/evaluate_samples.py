import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import pandas as pd
from scipy import stats, linalg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Generated Samples Quality (Academic LSTM Version)')
    parser.add_argument('--generated_path', type=str, required=False,
                        default=r'Data/output/Leak/ddpm_fake_Leak_major3_sub1.npy',
                        help='Path to generated samples .npy file')
    parser.add_argument('--original_path', type=str, required=False, 
                        default=r'Data/leak_signals/3/3',   
                        help='Path to original training samples directory or .npy file')
    parser.add_argument('--output_dir', type=str, default=r'Data/output/Leak/results_31',
                        help='Directory to save evaluation results')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--feature_idx', type=int, default=0,
                        help='Feature index to visualize')
    parser.add_argument('--lstm_epochs', type=int, default=30,
                        help='Training epochs for LSTM evaluators')
    return parser.parse_args()

def load_samples(path):
    """【严谨加载】绝不篡改物理量级，直接读取纯净数据"""
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
        data = np.load(path)
        # 注意：这里彻底移除了对 data 进行 (data+1)/2 的截断操作！
        # 传入的 .npy 必须是经由 dataset.scaler.inverse_transform 无损还原后的真实量级数据
        return data
    raise ValueError(f"Invalid path: {path}")

def load_samples_from_csv(data_root, use_cols=None, window=1780):
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
    print(f"Loaded {len(samples)} original samples from CSV files")
    return samples

# ========================================================
# 基础统计与分布指标计算
# ========================================================
def calculate_statistics(data):
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
    p_flat, q_flat = p.reshape(-1), q.reshape(-1)
    min_val, max_val = min(np.min(p_flat), np.min(q_flat)), max(np.max(p_flat), np.max(q_flat))
    p_hist, _ = np.histogram(p_flat, bins=bins, range=(min_val, max_val), density=False)
    q_hist, _ = np.histogram(q_flat, bins=bins, range=(min_val, max_val), density=False)
    p_hist = p_hist / np.sum(p_hist) + 1e-10
    q_hist = q_hist / np.sum(q_hist) + 1e-10
    m = 0.5 * (p_hist + q_hist)
    js_divergence = 0.5 * stats.entropy(p_hist, m) + 0.5 * stats.entropy(q_hist, m)
    return js_divergence

def calculate_wasserstein_distance(p, q):
    distances = []
    for i in range(p.shape[-1]):
        p_flat = p[:, :, i].reshape(-1)
        q_flat = q[:, :, i].reshape(-1)
        distances.append(stats.wasserstein_distance(p_flat, q_flat))
    return np.mean(distances)

def calculate_correlational_score(original, generated):
    print(' Calculating correlational score...')
    orig_corr = np.corrcoef(original.reshape(-1, original.shape[-1]).T)
    gen_corr = np.corrcoef(generated.reshape(-1, generated.shape[-1]).T)
    diff = np.abs(orig_corr - gen_corr).mean()
    return max(0.0, 1.0 - diff) * 100.0

def calculate_context_fid_score(original, generated):
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
    mu1, mu2 = np.mean(orig_feat, axis=0), np.mean(gen_feat, axis=0)
    sigma1, sigma2 = np.cov(orig_feat.T), np.cov(gen_feat.T)
    
    eps = 1e-6
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
    
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.dot(diff, diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return max(0.0, 100.0 * np.exp(-fid / 50.0))

# ========================================================
# 深度学习动力学评估核心 (LSTM)
# ========================================================
class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.linear(h_n[-1])
        return torch.sigmoid(out)

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)

def calculate_discriminative_score(original, generated, epochs=30):
    print(' Calculating Discriminative Score (Academic LSTM)...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = min(len(original), len(generated))
    
    X = np.vstack([original[:n_samples], generated[:n_samples]])
    Y = np.hstack([np.ones(n_samples), np.zeros(n_samples)])
    
    N, T, C = X.shape
    X_scaled = StandardScaler().fit_transform(X.reshape(-1, C)).reshape(N, T, C)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=32)
    
    model = LSTMDiscriminator(input_dim=C).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x.to(device)).squeeze()
            loss = criterion(out, batch_y.to(device))
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred = (model(batch_x.to(device)).squeeze() > 0.5).float()
            correct += (pred == batch_y.to(device)).sum().item()
            total += batch_y.size(0)
            
    accuracy = correct / total
    score = max(0.0, (1.0 - abs(accuracy - 0.5) * 2) * 100.0)
    print(f"  -> LSTM Classification Accuracy: {accuracy*100:.2f}%, Mapping to Score: {score:.2f}/100")
    return score

def calculate_predictive_score(original, generated, epochs=30):
    print(' Calculating Predictive Score (TSTR LSTM)...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = min(len(original), len(generated))
    
    N, T, C = generated[:n_samples].shape
    scaler = StandardScaler()
    gen_scaled = scaler.fit_transform(generated[:n_samples].reshape(-1, C)).reshape(N, T, C)
    orig_scaled = scaler.transform(original[:n_samples].reshape(-1, C)).reshape(N, T, C)
    
    gen_X, gen_Y = gen_scaled[:, :-1, :], gen_scaled[:, 1:, :]
    orig_X, orig_Y = orig_scaled[:, :-1, :], orig_scaled[:, 1:, :]
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(gen_X), torch.FloatTensor(gen_Y)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(orig_X), torch.FloatTensor(orig_Y)), batch_size=32)
    
    model = LSTMPredictor(input_dim=C).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x.to(device))
            loss = criterion(out, batch_y.to(device))
            loss.backward()
            optimizer.step()
            
    model.eval()
    total_mae = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred = model(batch_x.to(device))
            total_mae += criterion(pred, batch_y.to(device)).item() * batch_x.size(0)
            
    avg_mae = total_mae / len(test_loader.dataset)
    score = max(0.0, 100.0 * np.exp(-avg_mae))
    print(f"  -> TSTR Mean Absolute Error: {avg_mae:.4f}, Mapping to Score: {score:.2f}/100")
    return score

# ========================================================
# 绘图与报告生成
# ========================================================
def plot_statistics_comparison(original_stats, generated_stats, output_dir):
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
    sns.kdeplot(orig_feat, label='Original', fill=True)
    sns.kdeplot(gen_feat, label='Generated', fill=True)
    plt.xlabel('Value'); plt.ylabel('Density'); plt.title('KDE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'))
    plt.close()

def plot_sample_comparison(original, generated, output_dir, n_samples=5, feature_idx=0):
    os.makedirs(output_dir, exist_ok=True)
    orig_idx = np.random.choice(len(original), n_samples, replace=False)
    gen_idx = np.random.choice(len(generated), n_samples, replace=False)
    
    for i, (oi, gi) in enumerate(zip(orig_idx, gen_idx)):
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(original[oi, :, feature_idx], color='blue')
        plt.title(f'Original Sample {i}'); plt.xlabel('Time'); plt.ylabel(f'Feature {feature_idx}')
        plt.subplot(1, 2, 2)
        plt.plot(generated[gi, :, feature_idx], color='orange')
        plt.title(f'Generated Sample {i}'); plt.xlabel('Time'); plt.ylabel(f'Feature {feature_idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_comparison_{i}.png'))
        plt.close()

def generate_evaluation_report(original, generated, output_dir, lstm_epochs=30):
    os.makedirs(output_dir, exist_ok=True)
    
    orig_stats = calculate_statistics(original)
    gen_stats = calculate_statistics(generated)
    kl_div = calculate_kl_divergence(original, generated)
    wasserstein_dist = calculate_wasserstein_distance(original, generated)
    
    disc_score = calculate_discriminative_score(original, generated, epochs=lstm_epochs)
    pred_score = calculate_predictive_score(original, generated, epochs=lstm_epochs)
    corr_score = calculate_correlational_score(original, generated)
    fid_score = calculate_context_fid_score(original, generated)
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=== Advanced Sample Quality Evaluation Report (LSTM Version) ===\n\n')
        f.write(f'Original shape: {original.shape}\n')
        f.write(f'Generated shape: {generated.shape}\n\n')
        
        f.write('2. Statistical Metrics\n')
        for key in orig_stats:
            f.write(f'{key.upper()}:\n  Original: {orig_stats[key]}\n  Generated: {gen_stats[key]}\n')
        
        f.write(f'\n3. Distribution Metrics\n')
        f.write(f'KL Divergence: {kl_div:.4f}\n')
        f.write(f'Wasserstein Distance: {wasserstein_dist:.4f}\n')
        
        f.write(f'\n5. Advanced Physics & Dynamics Metrics\n')
        f.write(f'Discriminative Score (LSTM): {disc_score:.4f}/100\n')
        f.write(f'Predictive Score (TSTR): {pred_score:.4f}/100\n')
        f.write(f'Correlational Score: {corr_score:.4f}/100\n')
        f.write(f'Context FID Score: {fid_score:.4f}/100\n')
        
        stat_score = max(0, 100 - (kl_div * 10 + wasserstein_dist * 10))
        overall = (stat_score * 0.25 + disc_score * 0.25 + pred_score * 0.20 + 
                   corr_score * 0.15 + fid_score * 0.15)
        
        f.write(f'\n6. Quality Scores\n')
        f.write(f'Overall Quality Score: {overall:.2f}/100\n')
        
        if overall >= 80:
            f.write('\nConclusion: Excellent! Fully meets top-tier academic standards.\n')
        elif overall >= 60:
            f.write('\nConclusion: Good. Ready for downstream classification tasks.\n')
        elif overall >= 40:
            f.write('\nConclusion: Fair. Dynamics captured, but requires minor tuning.\n')
        else:
            f.write('\nConclusion: Poor. Check for numerical instability or OOD generation.\n')
    
    print(f'Report saved to: {report_path}')
    return overall

def main():
    args = parse_args()
    print(f'Loading samples...')
    
  
    
    args = parse_args()
    print(f'Loading samples...')
    
    original = load_samples(args.original_path)
    if np.isnan(original).any() or np.isinf(original).any():
        print("[WARNING] Original data contains NaN/Inf. Cleaning...")
        original = np.nan_to_num(original, nan=0.0, posinf=3.0, neginf=-3.0)
    # 【强制新增】：注入 1e-6 的极微小物理底噪，彻底打破方差为 0 导致的除零崩溃！
    original = original + np.random.normal(0, 1e-6, original.shape)
    
    print(f'Original samples shape: {original.shape}')
    
    generated = load_samples(args.generated_path)
    if np.isnan(generated).any() or np.isinf(generated).any():
        print("[WARNING] Generated data contains NaN/Inf. Cleaning...")
        generated = np.nan_to_num(generated, nan=0.0, posinf=3.0, neginf=-3.0)
    # 【强制新增】：注入 1e-6 的极微小物理底噪，拯救协方差矩阵和 FID 计算！
    generated = generated + np.random.normal(0, 1e-6, generated.shape)
    
    print(f'Generated samples shape: {generated.shape}')


    
    
    print(f'Creating output directory: {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print('Generating academic evaluation report...')
    overall_score = generate_evaluation_report(original, generated, args.output_dir, args.lstm_epochs)
    
    print('Plotting comparisons...')
    plot_statistics_comparison(calculate_statistics(original), calculate_statistics(generated), args.output_dir)
    plot_distribution_comparison(original, generated, args.output_dir, args.feature_idx)
    plot_sample_comparison(original, generated, args.output_dir, args.n_samples, args.feature_idx)
    
    print(f'\nEvaluation completed! Overall Score: {overall_score:.2f}/100')

if __name__ == '__main__':
    main()