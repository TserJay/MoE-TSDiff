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
        preload_to_memory=True  # 新增开关：控制是否预加载数据到内存
    
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


class HierarchicalLabelDataset(Dataset):
    """
    支持分层标签的数据集类
    数据目录结构: Data/leak_signals/{major_class}/{sub_class}/xxx.csv
    """
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
        use_cols=None,
        num_major_classes=4,
        num_sub_classes=12,
        preload_to_memory=True,  # 新增开关：控制是否预加载数据到内存
        **kwargs
    ):
        super(HierarchicalLabelDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        self.name = name
        self.window = window
        self.proportion = proportion
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one
        self.preload_to_memory = preload_to_memory  # 保存开关状态
        self.seed = seed
        self.period = period
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)
        
        self.num_major_classes = num_major_classes
        self.num_sub_classes = num_sub_classes
        
        self.all_samples, self.all_major_labels, self.all_sub_labels = self._read_all_data(data_root, use_cols)
        
        self.rawdata = np.concatenate(self.all_samples, axis=0)
        self.scaler = MinMaxScaler()
        self.scaler = self.scaler.fit(self.rawdata.reshape(-1, self.rawdata.shape[-1]))
        
        self.data = self._normalize(self.rawdata.reshape(-1, self.rawdata.shape[-1]))
        self.data = self.data.reshape(self.rawdata.shape)
        
        self.samples, self.major_labels, self.sub_labels = self._build_samples()
        
        train_idx, test_idx = self._divide_indices(len(self.samples), proportion, seed)
        
        if period == 'train':
            self.samples = self.samples[train_idx]
            self.major_labels = self.major_labels[train_idx]
            self.sub_labels = self.sub_labels[train_idx]
            self.training = True
        else:
            self.samples = self.samples[test_idx]
            self.major_labels = self.major_labels[test_idx]
            self.sub_labels = self.sub_labels[test_idx]
            self.training = False
        
        if self.preload_to_memory:
            print(f"预加载 {len(self.samples)} 个样本到内存中...")
            self.samples = torch.from_numpy(self.samples).float()
            self.major_labels = torch.from_numpy(self.major_labels).long()
            self.sub_labels = torch.from_numpy(self.sub_labels).long()
            self.sample_num = self.samples.shape[0]
            print(f"数据预加载完成，内存使用优化")
        else:
            # 不预加载，保存原始索引
            self.sample_num = len(self.samples)
            print(f"不预加载数据到内存，按需加载 {self.sample_num} 个样本")
            return  # 跳过后续预加载逻辑
        
        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_major_labels_{self.window}.npy"), self.major_labels.numpy())
            np.save(os.path.join(self.dir, f"{self.name}_sub_labels_{self.window}.npy"), self.sub_labels.numpy())

    def _read_all_data(self, data_root, use_cols):
        all_samples = []
        all_major_labels = []
        all_sub_labels = []
        
        for major_class in range(self.num_major_classes):
            major_dir = os.path.join(data_root, str(major_class))
            if not os.path.exists(major_dir):
                continue
                
            for sub_class in range(self.num_sub_classes):
                sub_dir = os.path.join(major_dir, str(sub_class))
                if not os.path.exists(sub_dir):
                    continue
                
                for filename in os.listdir(sub_dir):
                    if filename.endswith('.csv'):
                        filepath = os.path.join(sub_dir, filename)
                        df = pd.read_csv(filepath, header=0)
                        
                        if use_cols is not None:
                            data = df.iloc[:, use_cols].values
                        else:
                            data = df.iloc[:, [1, 3, 5]].values
                        
                        original_length = len(data)
                        
                        # 如果数据长度小于窗口大小，进行填充
                        if original_length < self.window:
                            print(f"填充文件 {filepath}: 数据长度 {original_length} 小于窗口大小 {self.window}，使用向前填充")
                            # 使用向前填充（用最后一个值填充）
                            pad_length = self.window - original_length
                            last_value = data[-1:, :]
                            padding = np.repeat(last_value, pad_length, axis=0)
                            data = np.vstack([data, padding])
                        
                        num_windows = len(data) - self.window + 1
                        all_samples.append(data)
                        all_major_labels.append(np.full(num_windows, major_class))
                        all_sub_labels.append(np.full(num_windows, sub_class))
        
        if len(all_samples) == 0:
            raise ValueError(f"没有找到符合条件的数据文件，请检查数据目录结构: {data_root}")
            
        return all_samples, np.concatenate(all_major_labels, axis=0), np.concatenate(all_sub_labels, axis=0)

    def _build_samples(self):
        samples = []
        label_idx = 0
        # 对归一化后的数据进行分段
        data_idx = 0
        for original_data in self.all_samples:
            num_windows = len(original_data) - self.window + 1
            for i in range(num_windows):
                samples.append(self.data[data_idx + i : data_idx + i + self.window, :])
                label_idx += 1
            data_idx += num_windows
        
        return np.array(samples), self.all_major_labels, self.all_sub_labels

    def _normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def _divide_indices(self, size, ratio, seed):
        st0 = np.random.get_state()
        np.random.seed(seed)
        
        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]
        
        np.random.set_state(st0)
        return regular_train_id, irregular_train_id

    def __getitem__(self, ind):
        if not self.preload_to_memory:
            # 不预加载时，从磁盘按需加载
            x = self.samples[ind, :, :]
            x = torch.from_numpy(x).float()
            
            # 训练时添加数据增强
            if self.period == 'train' and self.training:
                # 噪声注入 (σ=0.01-0.03)
                noise_level = 0.01 + 0.02 * torch.rand(1).item()
                noise = torch.randn_like(x) * noise_level
                x = x + noise
                
                # 幅度缩放 (0.8-1.2)
                scale = 0.8 + 0.4 * torch.rand(1).item()
                x = x * scale
                
                # 时间移位 (±5个时间步)
                shift = torch.randint(-5, 6, (1,)).item()
                if shift > 0:
                    x = torch.cat([x[shift:], x[:shift]], dim=0)
                elif shift < 0:
                    x = torch.cat([x[shift:], x[:shift]], dim=0)
                
                # 轻微的时间扭曲
                time_warp = 1.0 + 0.1 * (torch.rand(1).item() - 0.5)
                x = x * time_warp
            
            major_y = self.major_labels[ind]
            sub_y = self.sub_labels[ind]
            return x, major_y, sub_y
        
        x = self.samples[ind, :, :]
        
        # 训练时添加数据增强
        if self.period == 'train' and self.training:
            # 噪声注入 (σ=0.01-0.03)
            noise_level = 0.01 + 0.02 * torch.rand(1).item()
            noise = torch.randn_like(x) * noise_level
            x = x + noise
            
            # 幅度缩放 (0.8-1.2)
            scale = 0.8 + 0.4 * torch.rand(1).item()
            x = x * scale
            
            # 时间移位 (±5个时间步)
            shift = torch.randint(-5, 6, (1,)).item()
            if shift > 0:
                x = torch.cat([x[shift:], x[:shift]], dim=0)
            elif shift < 0:
                x = torch.cat([x[shift:], x[:shift]], dim=0)
            
            # 轻微的时间扭曲
            time_warp = 1.0 + 0.1 * (torch.rand(1).item() - 0.5)
            x = x * time_warp
        
        major_y = self.major_labels[ind]
        sub_y = self.sub_labels[ind]
        return x, major_y, sub_y

    def __len__(self):
        return self.sample_num
