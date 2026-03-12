# 生成特定类别数据功能说明

## 已完成的修改

### 1. main.py - 添加类别选择参数

**新增命令行参数**:
- `--major_class`: 指定大类标签（0-3）
- `--sub_class`: 指定小类标签（0-11）

**使用方法**:

```bash
# 生成所有类别的数据
python main.py --sample 1 --pred_num 100

# 生成大类3的数据
python main.py --sample 1 --pred_num 100 --major_class 3

# 生成大类3、小类2的数据
python main.py --sample 1 --pred_num 100 --major_class 3 --sub_class 2
```

**输出文件名**:
- 所有类别: `ddpm_fake_Leak.npy`
- 大类3: `ddpm_fake_Leak_major3.npy`
- 大类3小类2: `ddpm_fake_Leak_major3_sub2.npy`

### 2. trainer.sample() - 已支持类别参数

**函数签名**:
```python
def sample(self, num: int, size_every: int, shape: tuple = None, 
           model_kwargs: dict = None, cond_fn = None, major_y=None, sub_y=None)
```

**参数说明**:
- `major_y`: 大类标签（整数或None）
- `sub_y`: 小类标签（整数或None）

### 3. Diffusion_TS.generate_mts() - 已支持类别参数

**函数签名**:
```python
def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None, 
                 major_y=None, sub_y=None)
```

### 4. Diffusion_TS._train_loss() - 已支持类别参数

**训练时自动使用分层标签**:
```python
loss = self.model(data, target=data, major_y=major_y, sub_y=sub_y)
```

## 数据集支持

### HierarchicalLabelDataset

**数据结构**:
```
Data/leak_signals/
├── 0/              # 大类0
│   ├── 0/         # 小类0
│   │   ├── (1).csv
│   │   └── ...
│   └── 1/         # 小类1
│       ├── (1).csv
│       └── ...
├── 1/              # 大类1
│   └── ...
├── 2/              # 大类2
│   └── ...
└── 3/              # 大类3
    └── ...
```

**返回格式**:
```python
x, major_y, sub_y = dataset[ind]
# x: (2048, 3) - 信号数据
# major_y: int - 大类标签 (0-3)
# sub_y: int - 小类标签 (0-11)
```

## 训练流程

### 数据加载
```python
# build_dataloader返回的数据包含分层标签
batch = next(dataloader)
if isinstance(batch, (list, tuple)) and len(batch) == 3:
    data, major_y, sub_y = batch
```

### 模型训练
```python
# 自动使用分层标签
loss = model(data, target=data, major_y=major_y, sub_y=sub_y)
```

### 生成样本
```python
# 生成特定类别的样本
samples = trainer.sample(
    num=100, 
    size_every=50, 
    shape=[2048, 3],
    major_y=3,    # 大类3
    sub_y=2       # 小类2
)
```

## 评估流程

### 评估生成的特定类别数据
```bash
# 评估大类3小类2的数据
python tools/evaluate_samples.py \
    --generated_path "Data/output/Leak/ddpm_fake_Leak_major3_sub2.npy" \
    --original_path "Data/leak_signals/3/2"
```

### 评估所有数据
```bash
# 评估所有生成的数据
python tools/evaluate_samples.py \
    --generated_path "Data/output/Leak/ddpm_fake_Leak.npy" \
    --original_path "Data/leak_signals"
```

## 模型架构

### 分层条件编码

**嵌入层**:
```python
self.major_embedding = nn.Embedding(4, d_model)   # 4个大类
self.sub_embedding = nn.Embedding(12, d_model)    # 12个小类
```

**条件融合**:
```python
label_emb = self.major_embedding(major_y) + self.sub_embedding(sub_y)
```

**使用位置**:
- Encoder输入
- Decoder输入
- 注意力机制

## 使用示例

### 训练模型
```bash
python main.py --train
```

### 生成特定类别数据
```bash
# 生成大类3的所有数据
python main.py --sample 1 --pred_num 100 --major_class 3

# 生成大类3小类2的数据
python main.py --sample 1 --pred_num 100 --major_class 3 --sub_class 2
```

### 评估生成质量
```bash
# 评估大类3小类2的数据
python tools/evaluate_samples.py \
    --generated_path "Data/output/Leak/ddpm_fake_Leak_major3_sub2.npy" \
    --original_path "Data/leak_signals/3/2"
```

## 完整工作流程

1. **训练模型** (使用所有类别数据)
   ```bash
   python main.py --train
   ```

2. **生成特定类别数据**
   ```bash
   python main.py --sample 1 --pred_num 100 --major_class 3 --sub_class 2
   ```

3. **评估生成质量**
   ```bash
   python tools/evaluate_samples.py \
       --generated_path "Data/output/Leak/ddpm_fake_Leak_major3_sub2.npy" \
       --original_path "Data/leak_signals/3/2"
   ```

## 注意事项

1. **训练时**: 模型使用所有类别的数据进行训练
2. **生成时**: 可以指定特定类别生成
3. **评估时**: 应该使用对应类别的原始数据进行评估
4. **文件名**: 生成的文件名会自动根据类别设置

## 修改文件清单

1. `main.py` - 添加类别参数和文件名逻辑
2. `engine/solver.py` - sample()函数已支持类别参数
3. `Models/interpretable_diffusion/gaussian_diffusion.py` - generate_mts()已支持类别参数
4. `Utils/Data_utils/real_datasets.py` - HierarchicalLabelDataset已返回分层标签
