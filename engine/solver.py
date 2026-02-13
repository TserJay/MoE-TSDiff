import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.dataloader = dataloader['dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)
        
        # 添加混合精度训练配置
        self.use_amp = config['solver'].get('use_amp', True)
        if self.use_amp:
            # 使用新的API格式
            self.scaler = torch.amp.GradScaler()  # 添加混合精度训练的scaler
        else:
            self.scaler = None

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))
    
    def save_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current classifer to {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        data = {
            'step': self.step_classifier,
            'classifier': self.classifier.state_dict()
        }
        torch.save(data, str(self.results_folder / f'ckpt_classfier-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def load_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'ckpt_classfier-{milestone}.pt'), map_location=device)
        self.classifier.load_state_dict(data['classifier'])
        self.step_classifier = data['step']
        self.milestone_classifier = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        # 输出训练模式信息
        if self.logger is not None:
            train_mode = "混合精度 (AMP)" if self.use_amp else "单精度"
            self.logger.log_info(f'{self.args.name}: 开始{train_mode}训练...')
        
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    
                    if self.use_amp:
                        # 使用autocast上下文管理器进行混合精度训练
                        with torch.amp.autocast(device_type='cuda'):
                            loss = self.model(data, target=data)
                            loss = loss / self.gradient_accumulate_every
                        # 使用scaler进行梯度缩放
                        self.scaler.scale(loss).backward()
                    else:
                        # 使用单精度训练
                        loss = self.model(data, target=data)
                        loss = loss / self.gradient_accumulate_every
                        loss.backward()
                    
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                if self.use_amp:
                    # 使用scaler进行梯度裁剪
                    self.scaler.unscale_(self.opt)
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    # 使用scaler进行优化器步骤
                    self.scaler.step(self.opt)
                    self.sch.step(total_loss)
                    self.scaler.update()
                else:
                    # 使用单精度的梯度裁剪和优化器步骤
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.sch.step(total_loss)
                
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))
                    
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num: int, size_every: int, shape: tuple = None, model_kwargs: dict = None, cond_fn = None) -> np.ndarray:
        """
        从模型中生成新的样本
        
        Args:
            num: 生成样本总数
            size_every: 每次生成的样本数
            shape: 样本形状 [时间步数, 特征数]
            model_kwargs: 模型参数
            cond_fn: 条件函数
            
        Returns:
            生成的样本数组
        """
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for i in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every, model_kwargs=model_kwargs, cond_fn=cond_fn)
            samples = np.vstack([samples, sample.detach().cpu().numpy()])
            
            # 每处理几个批次后清理缓存
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        """
        从部分数据恢复完整数据
        
        Args:
            raw_dataloader: 原始数据加载器
            shape: 数据形状 [时间步数, 特征数]
            coef: 系数
            stepsize: 步长
            sampling_steps: 采样步数
            
        Returns:
            samples: 恢复的样本
            reals: 真实样本
            masks: 掩码
        """
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.vstack([samples, sample.detach().cpu().numpy()])
            reals = np.vstack([reals, x.detach().cpu().numpy()])
            masks = np.vstack([masks, t_m.detach().cpu().numpy()])
            
            # 每处理几个批次后清理缓存
            if (idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples

    def forward_sample(self, x_start):
       b, c, h = x_start.shape
       noise = torch.randn_like(x_start, device=self.device)
       t = torch.randint(0, self.model.num_timesteps, (b,), device=self.device).long()
       x_t = self.model.q_sample(x_start=x_start, t=t, noise=noise).detach()
       return x_t, t

    def train_classfier(self, classifier):
        device = self.device
        step = 0
        self.milestone_classifier = 0
        self.step_classifier = 0
        dataloader = self.dataloader
        dataloader.dataset.shift_period('test')
        dataloader = cycle(dataloader)

        self.classifier = classifier
        self.opt_classifier = Adam(filter(lambda p: p.requires_grad, self.classifier.parameters()), lr=5.0e-4)
        
        # 添加分类器训练的混合精度配置
        if self.use_amp:
            # 使用新的API格式
            self.scaler_classifier = torch.amp.GradScaler()  # 添加分类器训练的scaler
        else:
            self.scaler_classifier = None
        
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training classifier...'.format(self.args.name), check_primary=False)
        
        # 输出分类器训练模式信息
        if self.logger is not None:
            train_mode = "混合精度 (AMP)" if self.use_amp else "单精度"
            self.logger.log_info(f'{self.args.name}: 开始{train_mode}分类器训练...')
        
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    x, y = next(dataloader)
                    x, y = x.to(device), y.to(device)
                    x_t, t = self.forward_sample(x)
                    
                    if self.use_amp:
                        # 使用autocast上下文管理器进行混合精度训练
                        with torch.amp.autocast(device_type='cuda'):
                            logits = classifier(x_t, t)
                            loss = F.cross_entropy(logits, y)
                            loss = loss / self.gradient_accumulate_every
                        # 使用scaler进行梯度缩放
                        self.scaler_classifier.scale(loss).backward()
                    else:
                        # 使用单精度训练
                        logits = classifier(x_t, t)
                        loss = F.cross_entropy(logits, y)
                        loss = loss / self.gradient_accumulate_every
                        loss.backward()
                    
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                if self.use_amp:
                    # 使用scaler进行优化器步骤
                    self.scaler_classifier.step(self.opt_classifier)
                    self.scaler_classifier.update()
                else:
                    # 使用单精度的优化器步骤
                    self.opt_classifier.step()
                
                self.opt_classifier.zero_grad()
                self.step_classifier += 1
                step += 1

                with torch.no_grad():
                    if self.step_classifier != 0 and self.step_classifier % self.save_cycle == 0:
                        self.milestone_classifier += 1
                        self.save(self.milestone_classifier)
                                            
                    if self.logger is not None and self.step_classifier % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/classifier_loss', scalar_value=total_loss, global_step=self.step_classifier)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

        # return classifier

