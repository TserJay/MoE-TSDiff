# MoE-TSDiff: Hierarchical Mixture-of-Experts Diffusion for Industrial Time-Series Generation

> 在 Diffusion-TS（ICLR 2024）基础上深度改造：
> 引入 **Transformer MoE（混合专家）** + **层次化标签条件** + **动态退火 CFG**，
> 面向具有「孔径大类 / 工况小类」两级标签的工业时序信号（如管道泄漏检测信号）生成。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 相对原版 Diffusion-TS 的改动](#2-相对原版-diffusion-ts-的改动)
- [3. 核心架构](#3-核心架构)
- [4. 项目结构](#4-项目结构)
- [5. 环境安装](#5-环境安装)
- [6. 数据准备](#6-数据准备)
- [7. 快速开始](#7-快速开始)
- [8. 配置详解](#8-配置详解)
- [9. 关键模块说明](#9-关键模块说明)
- [10. 已知问题](#10-已知问题)
- [11. 后续路线图](#11-后续路线图)
- [12. 引用与致谢](#12-引用与致谢)

---

## 1. 项目简介

本项目的目标是**合成具有物理一致性的工业时序信号**，用于小样本工况下的数据增强。

原始 Diffusion-TS 面向通用时序（股票、电力、太阳辐照等）。本改造针对一个具体问题：
工业泄漏检测信号天然带有 **两级层次标签**——

- **大类（major）**：孔径规格，决定波形的整体轮廓（0 / 1 / 2 / 3 四种孔径）；
- **小类（sub）**：同一孔径下的不同工况，决定波形细节（2 ~ 11 号工况）。

若为每个「孔径 × 工况」组合各训一个独立模型，参数爆炸且小类样本饥饿。
本项目用 **共享专家 + 孔径专属专家** 的 MoE 结构：通用规律由共享专家学习、
孔径残差由专属专家学习，在 **≈12.5M 参数** 内兼顾「物理隔离性」与「参数效率」。

---

## 2. 相对原版 Diffusion-TS 的改动

| 维度 | 原版 Diffusion-TS | 本项目 |
|------|------------------|--------|
| 条件机制 | 单层类别标签 `label_emb` | **两级层次标签**：`major_emb + sub_emb` |
| FFN 结构 | 全层标准 FFN | 解码器**最后一层替换为 `ApertureMoE_FFN`**（1 共享 + N 专属） |
| 采样引导 | 无 / 固定 CFG | **动态退火 CFG**（scale 随扩散步线性衰减） |
| 训练损失 | L1 + Fourier loss | **L1 + 对数功率谱密度损失（Log-PSD）** |
| 数据加载 | `CustomDataset`（单文件滑窗） | **`HierarchicalLabelDataset`**（一文件一样本 + 目录即标签） |
| 无监督基线 | 无 | 训练时对小类嵌入做 **10% 置零**，供 CFG 使用 |

> 本项目已移除原版的分类器引导（classifier guidance）训练路径，改用 **Classifier-Free Guidance**。

---

## 3. 核心架构

### 3.1 孔径专家 MoE

**设计思想**：所有样本先过 **共享专家**（学通用流体力学 / 传感器规律），
再按孔径路由到 **专属专家**（只学该孔径相对通用规律的残差），最后归一化相加。

```
输入  x (B, T=1780, F=3), major_y (B,), sub_y (B,), t (B,)
        │
        ▼
┌──────────────────────────────────────────┐
│ 共享编码器 Encoder × n_layer_enc (=2)      │
│   输入: x + time_cond + label_emb         │
└──────────────────────────────────────────┘
        │  enc_cond（共享特征）
        ▼
┌──────────────────────────────────────────┐
│ 解码器 Decoder × n_layer_dec (=4)         │
│  Layer 0..2 : 标准 FFN                    │
│  Layer 3    : ApertureMoE_FFN  ← MoE 层   │
│     ├─ shared_expert   : 1 个共享专家      │
│     └─ specific_experts: 4 个孔径专属专家  │
└──────────────────────────────────────────┘
        │
        ▼
  out = Σ trend_i + Σ season_i   →  (B, T, F)
```

`ApertureMoE_FFN.forward`（`Models/interpretable_diffusion/transformer.py`）：

```python
shared_out = self.shared_expert(x)                     # 所有样本走共享专家
specific_out = torch.zeros_like(x)
for i in range(self.num_experts):                      # 按 major_y 硬路由
    mask = (major_y == i)
    if mask.any():
        specific_out[mask] = self.specific_experts[i](x[mask])
out = self.expert_norm(x + shared_out + specific_out)  # 残差 + LayerNorm
```

**要点**：
- **硬路由**：每个样本只激活一个专属专家（非 Top-k 软路由）；
- **参数效率**：`num_experts=4` 时专属专家合计约 2M 参数，远少于训练 4 个独立模型；
- **无条件样本**：`major_y == -1` 时取所有专属专家输出的平均（CFG 内部使用）。

### 3.2 动态退火 CFG

固定 guidance scale 的问题：大噪声阶段类别易混淆、小噪声阶段高频易失真。
本项目让 scale 随扩散步 **从大到小线性衰减**：

```python
# time 从 sampling_timesteps 降到 0
current_scale = 1.0 + (guidance_scale - 1.0) * (time / total_timesteps)

pred_noise_cond,   *_ = model_predictions(..., label_emb=label_emb_cond)    # 看小类标签
pred_noise_uncond, *_ = model_predictions(..., label_emb=label_emb_uncond)  # 小类清零
pred_noise = pred_noise_uncond + current_scale * (pred_noise_cond - pred_noise_uncond)
```

| 阶段 | time | current_scale | 作用 |
|------|------|---------------|------|
| 大噪声 | ≈ total_timesteps | ≈ guidance_scale | 强 CFG，拉开不同孔径轮廓 |
| 小噪声 | ≈ 0 | ≈ 1.0 | 弱 CFG，避免高频毛刺 |

**CFG 的两条轨道**：`major_emb`（大类）两轨共享、绝不置零；
仅 `sub_emb`（小类）在无条件轨道中清零——CFG 只放大「小类」区分度，
不破坏「大类」的物理轮廓。

### 3.3 时频联合物理损失

训练损失 = **时域 L1** + **对数功率谱密度损失（Log-PSD）**，
实现见 `Models/interpretable_diffusion/gaussian_diffusion.py` 的 `compute_advanced_physics_loss`：

```python
l1_loss = mean(|model_out - target|)
fft_out = rfft(model_out, dim=1, norm='ortho')   # 沿时间轴做 rFFT
fft_tgt = rfft(target,   dim=1, norm='ortho')
log_psd_loss = mean(|log(psd_out + eps) - log(psd_tgt + eps)|)
total_loss = l1_loss + 0.2 * log_psd_loss
```

- 沿 **时间轴 `dim=1`** 做 FFT（不是特征轴）；
- `norm='ortho'` 归一化，防止频域能量量级突变；
- 在对数空间约束，兼顾高频细节与梯度稳定。

---

## 4. 项目结构

```
MoE-TSDiff/
├── main.py                              # 主入口：训练 / 采样 / 补全 / 预测
├── requirements.txt
├── Config/
│   └── config.yaml                      # 主配置（MoE + 层次化标签）
├── Models/
│   └── interpretable_diffusion/
│       ├── gaussian_diffusion.py        # Diffusion_TS：前向/反向过程、损失、fast_sample
│       ├── transformer.py               # Encoder/Decoder/注意力/ApertureMoE_FFN/Fourier
│       ├── model_utils.py               # 归一化、位置编码、series_decomp 等
│       └── classifier.py                # 分类器（分类器引导用，当前主流程未启用）
├── engine/
│   ├── solver.py                        # Trainer：train()/sample()/restore()/load()/save()
│   ├── logger.py                        # TensorBoard + 文本日志
│   └── lr_sch.py                        # ReduceLROnPlateauWithWarmup 调度器
├── Data/
│   ├── build_dataloader.py              # build_dataloader() / build_dataloader_cond()
│   ├── leak_signals_preprocessed/       # 预处理后的训练数据（目录即标签）
│   ├── datasets/                        # 原版基准数据集（ETTh/energy/stocks/fMRI）
│   └── output/                          # 生成样本与真值 .npy（运行时产物）
├── Utils/
│   ├── io_utils.py                      # instantiate_from_config / load_yaml_config
│   ├── Data_utils/
│   │   ├── real_datasets.py             # HierarchicalLabelDataset（本项目在用）
│   │   ├── leak_signals_dataset.py      # 早期数据加载变体
│   │   └── ...                          # eeg / mujoco / sine 等原版数据集
│   └── masking_utils.py                 # infill 用的掩码生成
├── tools/                               # 预处理、评估、可视化脚本
├── figures/                             # 文档配图
└── ARCHITECTURE.md                      # 架构详解（含源码级说明）
```

---

## 5. 环境安装

```bash
# 推荐 Python 3.8 ~ 3.10
conda create -n moe-tsdiff python=3.8
conda activate moe-tsdiff

pip install -r requirements.txt
```

**核心依赖**（见 `requirements.txt`）：

| 依赖 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.0.1 | 模型与扩散过程 |
| `einops` | 0.6.0 | 张量重排 |
| `ema-pytorch` | 0.2.1 | 采样时的 EMA 模型 |
| `scikit-learn` | 1.1.2 | `MinMaxScaler` 归一化 |
| `scipy` | 1.8.1 | Savitzky-Golay 平滑、`.mat` 读取 |
| `pandas` / `numpy` | — | CSV 加载与数组运算 |
| `pyyaml` | 6.0 | 配置加载 |
| `tqdm` | 4.64.1 | 进度条 |

> **注意**：`engine/logger.py` 依赖 `tensorboard`，但当前 `requirements.txt` 未列出，
> 请额外 `pip install tensorboard`，否则只会有文本日志。
> `requirements.txt` 中的 `mujoco` / `dm-control` / `gluonts` 属于原版遗留，本实验不需要。

---

## 6. 数据准备

### 6.1 目录结构

数据采用 **「一个 CSV = 一个独立样本，目录名 = 标签」** 的组织方式：

```
Data/leak_signals_preprocessed/
├── 0/                    # 孔径大类 0
│   ├── 2/  ├── 3/  ...   # 工况小类 2 ~ 11
│   │   ├── (1).csv
│   │   └── ...
├── 1/                    # 孔径大类 1
├── 2/                    # 孔径大类 2
└── 3/                    # 孔径大类 3
```

当前数据集的实际分布（实测）：

| 大类 | 样本数 | 存在的小类目录 |
|------|--------|----------------|
| 0 | 450 | 2 ~ 11 |
| 1 | 449 | 2 ~ 11 |
| 2 | 449 | 2 ~ 11 |
| 3 | 449 | 2 ~ 11 |
| **合计** | **1797** | — |

### 6.2 数据格式要求

- 每个 CSV 为**单个时序样本**，默认 1780 个时间点 + 1 行表头（共 1781 行）；
- 表头形如 `时间 - 电压, 幅值 - 电压, 时间 - 电压_0, 幅值 - 电压_0, ...`；
- 通过 `use_cols` 选取实际使用的列（默认 `[1, 3, 5]`，即第 2/4/6 列）；
- 加载时自动：
  - **长度对齐**：超长截断到 `window`，不足则用最后一个稳态值向后 padding；
  - **全局归一化**：`MinMaxScaler` 拟合后映射到 `[-1, 1]`（diffusion 友好区间）；
  - **随机划分**：按 `seed` 打乱后按 `proportion` 切分 train / test。

---

## 7. 快速开始

### 7.1 训练

```bash
python main.py \
  --name Leak \
  --config_file Config/config.yaml \
  --gpu 0 \
  --train
```

训练日志与检查点保存到 `config.yaml` 中 `solver.results_folder` 指定的目录。

**训练关键机制**：

- **10% 蒙眼盲打**：`sub_emb` 有 10% 概率被置零，强制模型学习无条件基线，为 CFG 提供条件；
- **大类不动**：`major_y` 绝不 mask，保证 MoE 路由始终正确；
- **EMA**：指数移动平均（`decay=0.99`）平滑参数，采样时使用 EMA 模型；
- **学习率调度**：Warmup（400 步）+ ReduceLROnPlateau；
- **NaN 防护**：loss / 梯度出现 NaN/Inf 时跳过该步。

### 7.2 条件生成

```bash
# 生成「大类 2 + 小类 5」的样本
python main.py \
  --name Leak \
  --config_file Config/config.yaml \
  --gpu 0 \
  --milestone 10 \
  --pred_num 100 \
  --major_class 2 \
  --sub_class 5
```

输出：`Data/output/Leak/ddpm_fake_Leak_major2_sub5.npy`（形状 `(100, 1780, 3)`）。

| 参数 | 含义 |
|------|------|
| `--milestone N` | 加载第 N 个检查点（`checkpoint-N.pt`） |
| `--pred_num N` | 生成样本数量 |
| `--major_class` | 孔径大类（0~3） |
| `--sub_class` | 工况小类（2~11） |

### 7.3 条件补全与预测

```bash
# 缺失值填补（infill）
python main.py --name Leak --config_file Config/config.yaml --gpu 0 \
  --sample 1 --mode infill --milestone 10 --missing_ratio 0.2

# 预测（predict）
python main.py --name Leak --config_file Config/config.yaml --gpu 0 \
  --sample 1 --mode predict --milestone 10 --pred_len 1780
```

---

## 8. 配置详解

`Config/config.yaml` 的核心字段：

```yaml
model:
  target: Models.interpretable_diffusion.gaussian_diffusion.Diffusion_TS
  params:
    seq_length: 1780            # 时序长度
    feature_size: 3             # 特征维度
    num_major_classes: 4        # 大类数（对应专属专家数）
    num_sub_classes: 12         # 小类数（实际数据用到 2~11）

    n_layer_enc: 2              # 编码器层数
    n_layer_dec: 4              # 解码器层数
    d_model: 64                 # 模型维度
    n_heads: 4                  # 注意力头数
    timesteps: 2000             # 训练扩散步数
    sampling_timesteps: 1000    # 采样步数（DDIM 加速）
    loss_type: 'l1'
    beta_schedule: 'cosine'

    use_moe: True               # 启用 MoE
    num_experts: 4              # 专属专家数（= 大类数）
    moe_layers: [3]             # 仅解码器第 4 层（索引 3）启用 MoE

solver:
  base_lr: 1.0e-4
  max_epochs: 50000
  save_cycle: 5000              # 每 5000 步保存检查点
  use_amp: False                # 混合精度开关
  ema: { decay: 0.99, update_interval: 10 }
  scheduler:
    target: engine.lr_sch.ReduceLROnPlateauWithWarmup
    params: { warmup: 400, patience: 50, factor: 0.5 }

dataloader:
  train_dataset:
    target: Utils.Data_utils.real_datasets.HierarchicalLabelDataset
    params:
      data_root: Data/leak_signals_preprocessed   # ⚠ 见「已知问题」第 1 条
      window: 1780
      proportion: 0.8
      use_cols: [1, 3, 5]
      num_major_classes: 4
      num_sub_classes: 12
      neg_one_to_one: True
      preload_to_memory: True
  batch_size: 16
```

---

## 9. 关键模块说明

### 9.1 `ApertureMoE_FFN` — `Models/interpretable_diffusion/transformer.py`

- `shared_expert`：`Linear → GELU → Dropout → Linear → Dropout`
- `specific_experts`：`ModuleList`，长度为 `num_experts`
- `expert_norm`：`LayerNorm(d_model)`，稳定各专家输出量级
- 无条件样本（`major_y == -1`）：取所有专属专家输出的平均

### 9.2 `fast_sample` 与动态退火 CFG — `Models/interpretable_diffusion/gaussian_diffusion.py`

推断阶段双轨预测：

1. **条件轨道**：`label_emb_cond = major_emb + sub_emb_cond`
2. **无条件轨道**：`label_emb_uncond = major_emb + 0`
3. **退火融合**：`current_scale` 随时间递减，融合两条轨道的噪声预测

### 9.3 `HierarchicalLabelDataset` — `Utils/Data_utils/real_datasets.py`

按目录结构（`{major}/{sub}/*.csv`）自动生成 `major_y` / `sub_y`，并完成：

- 长度对齐（截断 / padding 到 `window`）
- 列选择（`use_cols`）
- 全局 MinMax 归一化到 `[-1, 1]`
- 可选内存预加载

---

## 10. 已知问题

> 本节记录审查中**已实测确认**的问题，供后续修复跟踪。README 正文按「应有状态」描述，
> 以下条目为当前代码与文档的实际差异。

| # | 严重度 | 位置 | 问题 |
|---|--------|------|------|
| 1 | **P0** | `Config/config.yaml:68,86` | `data_root` 写成 `Data/leak_signals_preprocessed/leak_signals_preprocessed`（重复嵌套），该路径**不存在**，会导致数据集构建为空并崩溃。应改为 `Data/leak_signals_preprocessed`。 |
| 2 | **P0** | `Utils/Data_utils/real_datasets.py:307-313` | train / test 划分**重叠 60%**：test 的 `proportion: 0.2` 被当作 `train_size`，两数据集同 seed 产生相同排列，test 取到 train 的子集。测试集严重泄漏，评估指标不可信。 |
| 3 | **P0** | `main.py:105` ↔ `real_datasets.py` | `HierarchicalLabelDataset` 未定义 `self.var_num`，但 infill / predict 路径读取 `dataset.var_num` → `AttributeError`。 |
| 4 | **P0** | `transformer.py:347` | `major_y=None` 时 `(major_y == -1)` 得到 Python `bool`，`.any()` 抛 `AttributeError`。无条件生成会命中。 |
| 5 | P1 | `engine/solver.py:152,389,412` | 使用 `torch.amp.*`（torch **2.1+** API），而 `requirements.txt` 锁 `torch==2.0.1`。当前 `use_amp: False` 掩盖，开启 AMP 即崩。 |
| 6 | P1 | `engine/logger.py:84` | tensorboard 不可用时 `tb_writer=None`，`close()` 抛 `AttributeError`。 |
| 7 | P1 | `main.py:31` | `--gpu` 默认值 `1`，单卡机器上 `set_device(1)` 失败。 |
| 8 | P2 | `gaussian_diffusion.py:15` | `compute_diff_loss` 及 `use_diff` / `diff_weight` 定义了但从未调用——文档中「L_diff」实为不存在的损失项。 |
| 9 | P2 | `gaussian_diffusion.py:129-136` | `use_ff` / `ff_weight` / `reg_weight` 均未被使用，PSD 权重硬编码为 `0.2`。 |
| 10 | P2 | 仓库卫生 | 32 个 `.pyc` 被跟踪；`.gitignore` 自身未被跟踪；`tools/` 目录忽略策略自相矛盾。 |

---

## 11. 后续路线图

### 11.1 架构
- [ ] **稀疏软路由**：Top-k 软路由 + 负载均衡损失（Switch Transformer 风格）
- [ ] **多层级 MoE**：小类也引入专家，形成 `major-expert → sub-expert` 两级结构
- [ ] **专家数自适应**：按数据分布动态决定专家数

### 11.2 训练
- [ ] **课程学习 CFG**：训练初期低 guidance_scale，后期渐增
- [ ] **专家 Dropout**：随机屏蔽专属专家，增强共享专家鲁棒性
- [ ] **对比学习损失**：潜空间同类拉近、异类推远

### 11.3 推断
- [ ] **无条件轨道缓存**：无条件分支不依赖小类标签，可缓存中间结果加速
- [ ] **每类 guidance_scale 搜索**：按大类自动搜索最优 scale

### 11.4 数据与评估
- [ ] **少数类增强**：jittering / scaling / time-warp
- [ ] **强化物理约束**：频域 PSD 约束、微分平滑约束
- [ ] **可视化评估**：t-SNE 潜空间分布、频谱对比、自相关对比

### 11.5 工程化
- [ ] **AMP 数值稳定性**：升级 torch 至 2.1+ 并启用 FP16
- [ ] **DDP 多卡训练**：支持更大 batch
- [ ] **ONNX / TensorRT 导出**：面向边缘部署

---

## 12. 引用与致谢

本项目基于 **Diffusion-TS**（ICLR 2024）二次开发：

```bibtex
@inproceedings{yuan2024diffusionts,
  title={Diffusion-{TS}: Interpretable Diffusion for General Time Series Generation},
  author={Xinyu Yuan and Yan Qiao},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=4h1apFjO99}
}
```

MoE 架构、层次化条件与动态退火 CFG 为本项目定制扩展。

同时致谢原版所借鉴的开源工作：
denoising-diffusion-pytorch、
VQ-Diffusion、
Diffusion-LM、
[N-BEATS](https://github.com/philipperemy/n-beats)、
ETSformer、
CSDI、
TimeGAN。

本仓库采用 **MIT License**（见 `LICENSE`）。
