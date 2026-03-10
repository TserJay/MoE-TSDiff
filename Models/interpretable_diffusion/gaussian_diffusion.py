import math
import torch
import torch.nn.functional as F
import torch.fft as fft

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract


def compute_diff_loss(x, y):
    """
    Compute differential loss between two sequences.
    Uses finite differences to approximate derivatives and computes MSE between derivatives.
    包含一阶和二阶微分损失，增强时间序列的平滑性
    """
    if x.shape[-1] < 2:
        return torch.tensor(0.0, device=x.device)
    
    # 一阶微分（速度）- 保证连续性
    dx = torch.diff(x, dim=-1)
    dy = torch.diff(y, dim=-1)
    first_order_loss = F.mse_loss(dx, dy, reduction='none')
    
    # 二阶微分（加速度）- 保证平滑性
    if x.shape[-1] >= 3:
        ddx = torch.diff(dx, dim=-1)
        ddy = torch.diff(dy, dim=-1)
        second_order_loss = F.mse_loss(ddx, ddy, reduction='none')
        return (first_order_loss.mean() + second_order_loss.mean()) / 2
    else:
        return first_order_loss.mean()


def compute_advanced_physics_loss(model_out, target, use_psd=True):
    """
    数值稳定的时频联合物理损失函数 (移除了极易导致NaN的四阶矩算子)
    model_out: 预测的原始信号 \hat{x}_0
    target: 真实的原始信号 x_0
    """
    eps = 1e-8
    total_loss = 0.0
    
    # 1. 基础时域 L1 损失
    l1_loss = torch.mean(torch.abs(model_out - target))
    total_loss += l1_loss
    
    # 2. 对数功率谱密度损失 (Log-PSD Loss)
    log_psd_loss = torch.tensor(0.0, device=model_out.device)
    if use_psd:
        # 使用 ortho 归一化防止频域能量量级突变
        fft_out = fft.rfft(model_out, dim=1, norm='ortho')
        fft_tgt = fft.rfft(target, dim=1, norm='ortho')
        
        psd_out = torch.real(fft_out)**2 + torch.imag(fft_out)**2
        psd_tgt = torch.real(fft_tgt)**2 + torch.imag(fft_tgt)**2
        
        # 对数空间约束高频，L1距离防止梯度爆炸
        log_psd_loss = torch.mean(torch.abs(torch.log(psd_out + eps) - torch.log(psd_tgt + eps)))
        
        # 严格限制频域权重，确保其作为辅助损失
        total_loss += 0.2 * log_psd_loss
        
        # 打印原始损失绝对值，用于调试
        if model_out.requires_grad and torch.is_grad_enabled():
            print(f"Raw L_time: {l1_loss.item():.6f} | Raw L_freq: {log_psd_loss.item():.6f}")
    
    return total_loss, l1_loss, log_psd_loss


# gaussian diffusion trainer class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=False,
            reg_weight=None,
            num_major_classes=4,
            num_sub_classes=12,
            **kwargs
    ):
        super(Diffusion_TS, self).__init__()

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        # 动态计算傅里叶损失权重，根据序列长度自适应调整
        if reg_weight is not None:
            self.ff_weight = reg_weight
        else:
            self.ff_weight = math.sqrt(self.seq_length) / 5  # 动态计算权重，根据序列长度自适应调整
        
        # 微分损失权重
        self.use_diff = kwargs.get('use_diff', False)  # 默认禁用微分损失
        self.diff_weight = kwargs.get('diff_weight', 1.0)

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

        # 分层类别 Embedding 层
        self.num_major_classes = num_major_classes
        self.num_sub_classes = num_sub_classes

        # 分别创建大类和小类的特征向量表
        self.major_embedding = nn.Embedding(self.num_major_classes, d_model)
        self.sub_embedding = nn.Embedding(self.num_sub_classes, d_model)


    def predict_noise_from_start(self, x_t, t, x0):
        sqrt_recipm1 = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        # 添加安全检查，避免除以零
        sqrt_recipm1 = torch.clamp(sqrt_recipm1, min=1.0e-10)
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                sqrt_recipm1
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, t, padding_masks=None, label_emb=None):
        trend, season = self.model(x, t, padding_masks=padding_masks, label_emb=label_emb)
        model_output = trend + season
        # 移除裁剪，让模型自由预测
        # model_output = torch.clamp(model_output, min=-1., max=1.)
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None, label_emb=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, padding_masks, label_emb=label_emb)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True, label_emb=None):
        _, x_start = self.model_predictions(x, t, label_emb=label_emb)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    def p_sample(self, x, t: int, clip_denoised=True, cond_fn=None, model_kwargs=None, label_emb=None):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised, label_emb=label_emb)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        if cond_fn is not None:
            model_mean = self.condition_mean(
                cond_fn, model_mean, model_log_variance, x, t=batched_times, model_kwargs=model_kwargs
            )
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    @torch.no_grad()
    def sample(self, shape, clip_denoised=True):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t, clip_denoised=clip_denoised)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, major_y=None, sub_y=None, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 
        img = torch.randn(shape, device=device)

        # 在生成开始前，计算出你要生成的联合目标特征
        label_emb = None
        if major_y is not None and sub_y is not None:
            label_emb = self.major_embedding(major_y) + self.sub_embedding(sub_y)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            
            # 携带联合特征 label_emb 进行条件定向预测
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised, label_emb=label_emb)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img
    
    def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None, major_y=None, sub_y=None, clip_denoised=True):
        feature_size, seq_length = self.feature_size, self.seq_length
        if cond_fn is not None:
            sample_fn = self.fast_sample_cond if self.fast_sampling else self.sample_cond
            return sample_fn((batch_size, seq_length, feature_size), model_kwargs=model_kwargs, cond_fn=cond_fn, clip_denoised=clip_denoised)
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        if major_y is not None and sub_y is not None:
            return sample_fn((batch_size, seq_length, feature_size), major_y=major_y, sub_y=sub_y, clip_denoised=clip_denoised)
        return sample_fn((batch_size, seq_length, feature_size), clip_denoised=clip_denoised)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, x, major_y=None, sub_y=None, **kwargs): 
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss_with_components(x_start=x, t=t, major_y=major_y, sub_y=sub_y, **kwargs)
    
    def _train_loss_with_components(self, x_start, t, major_y=None, sub_y=None, target=None, noise=None, padding_masks=None):
        """返回loss分解信息，用于tensorboard记录"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start
        x = self.q_sample(x_start=x_start, t=t, noise=noise) 

        # 计算联合的 label_emb (大类特征 + 小类特征)
        label_emb = None
        if major_y is not None and sub_y is not None:
            label_emb = self.major_embedding(major_y) + self.sub_embedding(sub_y)

        # 把融合后的 label_emb 传给底层预测网络
        model_out = self.output(x, t, padding_masks, label_emb=label_emb)

        # 使用高级物理损失函数
        total_loss, l1_loss, log_psd_loss = compute_advanced_physics_loss(model_out, target)

        # 检查loss是否为NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"警告: 发现NaN/Inf loss，model_out范围: [{model_out.min():.4f}, {model_out.max():.4f}], target范围: [{target.min():.4f}, {target.max():.4f}]")
            total_loss = torch.tensor(0.1, device=total_loss.device)
        
        # 返回总loss和各个组件
        loss_dict = {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'log_psd_loss': log_psd_loss
        }
        
        return total_loss, loss_dict

    def _train_loss(self, x_start, t, major_y=None, sub_y=None, target=None, noise=None, padding_masks=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start
        x = self.q_sample(x_start=x_start, t=t, noise=noise) 

        # 计算联合的 label_emb (大类特征 + 小类特征)
        label_emb = None
        if major_y is not None and sub_y is not None:
            label_emb = self.major_embedding(major_y) + self.sub_embedding(sub_y)

        # 把融合后的 label_emb 传给底层预测网络
        model_out = self.output(x, t, padding_masks, label_emb=label_emb)

        # 使用高级物理损失函数
        total_loss, _, _ = compute_advanced_physics_loss(model_out, target)

        # 添加微分损失（适度权重，平衡平滑性和细节保留）
        diff_loss = compute_diff_loss(model_out, target)
        if self.use_diff:
            total_loss += self.diff_weight * diff_loss  # 使用配置的微分权重
        
        # 检查loss是否为NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"警告: 发现NaN loss，model_out范围: [{model_out.min():.4f}, {model_out.max():.4f}], target范围: [{target.min():.4f}, {target.max():.4f}]")
            total_loss = torch.tensor(0.1, device=total_loss.device)
        
        return total_loss

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        img[partial_mask] = target[partial_mask]
        return img
    
    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        
        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    
    def condition_mean(self, cond_fn, mean, log_variance, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x=x, t=t, **model_kwargs)
        new_mean = (
            mean.float() + torch.exp(log_variance) * gradient.float()
        )
        return new_mean
    
    def condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)

        eps = self.predict_noise_from_start(x, t, x_start)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        pred_xstart = self.predict_start_from_noise(x, t, eps)
        model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, pred_xstart
    
    def sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(img, t, clip_denoised=clip_denoised, cond_fn=cond_fn,
                                         model_kwargs=model_kwargs)
        return img

    def fast_sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if cond_fn is not None:
                _, x_start = self.condition_score(cond_fn, x_start, img, time_cond, model_kwargs=model_kwargs)
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img


if __name__ == '__main__':
    pass
