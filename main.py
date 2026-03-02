import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config



def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    
    parser.add_argument('--name', type=str, default='Leak')
    parser.add_argument('--config_file', type=str, default=r'Config/config.yaml', 
                        help='path of config file')
    parser.add_argument('--output', type=str, default=r'Data/output', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='predict',
                        help='infill or predict.')
    parser.add_argument('--milestone', type=int, default=8)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_num',type=int, default=10, help='Number of Predictions.')
    parser.add_argument('--pred_len', type=int, default=2048, help='Length of Predictions.')
    
    # args for generating specific classes
    parser.add_argument('--major_class', type=int, default=3,
                        help='Major class to generate (None for all classes)')
    parser.add_argument('--sub_class', type=int, default=0,
                        help='Sub class to generate (None for all classes)')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    
    # 如果是infill或predict模式，需要加载条件数据集
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
        dataloader_info = None  # 避免加载普通训练数据集
    elif args.train:
        # 训练模式，需要加载训练数据集
        dataloader_info = build_dataloader(config, args)
    else:
        # 生成模式，不需要加载数据集
        dataloader_info = None

    # x: torch.FloatTensor, shape = (batch_size, channels, seq_len)
    # y: torch.LongTensor, shape = (batch_size,)- 可能是 label_pos 或 label_cls


    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}.npy'), samples)
    else:
        # 普通生成模式，不需要加载数据集
        trainer.load(args.milestone)
        
        # 从config中获取模型参数
        seq_length = config['model']['params']['seq_length']
        feature_size = config['model']['params']['feature_size']
        
        # 如果指定了类别，生成特定类别的数据
        major_y = args.major_class
        sub_y = args.sub_class
        
        samples = trainer.sample(num=args.pred_num, size_every=50, shape=[seq_length, feature_size], 
                                major_y=major_y, sub_y=sub_y)
        
        # 根据是否指定类别，设置不同的文件名
        if major_y is not None and sub_y is not None:
            filename = f'ddpm_fake_{args.name}_major{major_y}_sub{sub_y}.npy'
        elif major_y is not None:  
            filename = f'ddpm_fake_{args.name}_major{major_y}.npy'
        else:
            filename = f'ddpm_fake_{args.name}.npy'
            
        np.save(os.path.join(args.save_dir, filename), samples)
        print(f'Samples saved to: {os.path.join(args.save_dir, filename)}')

if __name__ == '__main__':
    main()
  
