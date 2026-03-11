import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3,
        preload_to_memory=True,  # 新增开关：控制是否预加载数据到内存
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one
        self.preload_to_memory = preload_to_memory  # 保存开关状态

        self.data = self.__normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)

        if self.preload_to_memory:
            self.samples = train if period == 'train' else inference
            print(f"预加载 {self.samples.shape[0]} 个样本到内存中...")
            self.samples = torch.from_numpy(self.samples).float()
        else:
            # 不预加载，保存原始索引和数据路径
            self.train_data = train if period == 'train' else inference
            self.sample_num = self.train_data.shape[0]
            print(f"不预加载数据到内存，按需加载 {self.sample_num} 个样本")
            return  # 跳过后续预加载逻辑
        
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
            self.masking = torch.from_numpy(self.masking)
        self.sample_num = self.samples.shape[0]
        print(f"数据预加载完成，内存使用优化")

    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)

        data = df.iloc[1:,[1,3,5]]
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
    
    def mask_data(self, seed=2023):
        if not self.preload_to_memory:
            raise RuntimeError("mask_data 仅在预加载模式下可用，请设置 preload_to_memory=True")
        masks = np.ones_like(self.samples)
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if not self.preload_to_memory:
            # 不预加载时，从磁盘按需加载
            x = self.train_data[ind, :, :]
            x = torch.from_numpy(x).float()
            return x
        
        if self.period == 'test':
            x = self.samples[ind, :, :]
            m = self.masking[ind, :, :]
            return x, m
        x = self.samples[ind, :, :]
        return x

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler


import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class HierarchicalLabelDataset(Dataset):
    """
    终极纯净版：支持分层标签的数据集类
    绝对遵循：1 个 CSV = 1 个独立泄漏事件 = 1780 个连续时间点
    数据目录结构: Data/leak_signals/{major_class}/{sub_class}/xxx.csv
    """
    def __init__(
        self, 
        name,
        data_root, 
        window=1780, 
        proportion=0.8, 
        save2npy=False, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        use_cols=[1, 3, 5],
        num_major_classes=4,
        num_sub_classes=12,
        preload_to_memory=True,  # 现在的逻辑非常轻量，强烈建议预加载！
        **kwargs
    ):
        super().__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        self.name = name
        self.window = window
        self.period = period
        self.auto_norm = neg_one_to_one
        
        samples_list = []
        major_labels_list = []
        sub_labels_list = []
        
        print(f"[{period.upper()}] 开始从 {data_root} 读取物理泄漏文件...")
        
        # 1. 独立读取每一个 CSV 作为绝对独立的样本
        for major_class in range(num_major_classes):
            major_dir = os.path.join(data_root, str(major_class))
            if not os.path.exists(major_dir): 
                continue
                
            for sub_class in range(num_sub_classes):
                sub_dir = os.path.join(major_dir, str(sub_class))
                if not os.path.exists(sub_dir): 
                    continue
                    
                for filename in os.listdir(sub_dir):
                    if not filename.endswith('.csv'): 
                        continue
                    
                    filepath = os.path.join(sub_dir, filename)
                    df = pd.read_csv(filepath, header=0)
                    
                    # 提取指定的 3 个通道
                    data = df.iloc[:, use_cols].values
                    
                    # =======================================================
                    # 【核心强制对齐】：让每个文件死死锚定在 1780 个时间点
                    # =======================================================
                    if len(data) >= self.window:
                        # 如果文件太长，只截取发生泄漏的前 1780 个关键点
                        data = data[:self.window, :]
                    else:
                        # 如果文件不到 1780，用最后一个稳态值向后平移填充（Pad）
                        pad_length = self.window - len(data)
                        last_val = data[-1:]
                        padding = np.repeat(last_val, pad_length, axis=0)
                        data = np.vstack([data, padding])
                        
                    samples_list.append(data)
                    major_labels_list.append(major_class)
                    sub_labels_list.append(sub_class)
                    
        # 2. 组装全局数组：此时必定是极其完美的 (N, 1780, 3)
        self.raw_samples = np.array(samples_list)
        self.major_labels = np.array(major_labels_list)
        self.sub_labels = np.array(sub_labels_list)
        
        N, L, C = self.raw_samples.shape
        print(f"成功读取！共获取 {N} 个独立文件样本，Shape: {self.raw_samples.shape}")

        # 3. 完美归一化 (Scaler)
        self.scaler = MinMaxScaler()
        # 展平为 (N*1780, 3) 计算全局极值，保证物理比例不畸变
        flat_data = self.raw_samples.reshape(-1, C)
        norm_data = self.scaler.fit_transform(flat_data)
        
        if self.auto_norm:
            # Diffusion 最喜欢的 [-1, 1] 区间
            norm_data = norm_data * 2.0 - 1.0
            
        self.all_samples = norm_data.reshape(N, L, C)
        
        # 4. 划分训练/测试集（使用随机打乱，保证分布均匀）
        np.random.seed(seed)
        indices = np.random.permutation(N)
        train_size = int(N * proportion)
        
        if period == 'train':
            valid_idx = indices[:train_size]
        else:
            valid_idx = indices[train_size:]
            
        # 直接转化为 Tensor 常驻内存，速度起飞！
        self.samples = torch.from_numpy(self.all_samples[valid_idx]).float()
        self.major_y = torch.from_numpy(self.major_labels[valid_idx]).long()
        self.sub_y = torch.from_numpy(self.sub_labels[valid_idx]).long()
        
        print(f"[{period.upper()}] 数据集构建完毕！分配了 {len(self.samples)} 个样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 【终极防线】：这里绝对、绝对不要加任何数据增强！
        # 自然物理信号的波峰位置一旦被 shift 或者扭曲，网络就会彻底精神分裂！
        return self.samples[idx], self.major_y[idx], self.sub_y[idx]
