import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class MyCSVDataset(Dataset):
    def __init__(self, data_root, window=1024, period="train", use_cols=[1,3,5]):
        """
        data_root: 数据集路径 (目录或者单个 csv 文件)
        window: 信号长度 (默认 1024)
        period: train / val / test
        use_cols: 选择 csv 的哪些列作为特征
        """
        self.data_root = data_root
        self.window = window
        self.period = period
        self.use_cols = use_cols

        self.samples = []
        self.labels = []

        # 如果是目录，批量读取
        if os.path.isdir(data_root):
            file_list = os.listdir(data_root)
            for file in file_list:
                df = pd.read_csv(os.path.join(data_root, file))
                x = df.values[:, use_cols][:window].astype(float)
                y = int(file.split("_")[0])  # 假设文件名里有类别，比如 "0_xxx.csv"
                self.samples.append(x)
                self.labels.append(y)
        else:  # 单文件
            df = pd.read_csv(data_root)
            x = df.values[:, use_cols][:window].astype(float)
            self.samples.append(x)
            self.labels.append(0)  # 单文件先默认 0 类

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
