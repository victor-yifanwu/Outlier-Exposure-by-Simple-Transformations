import os
import logging
import numpy as np
import torch
import random

def set_seed(seed):
    # 设置 Python 内置的随机数生成器种子
    random.seed(seed)
    
    # 设置 NumPy 随机数生成器种子
    np.random.seed(seed)
    
    # 设置 PyTorch 随机数生成器种子
    torch.manual_seed(seed)
    
    # 如果使用的是 GPU，设置所有 GPU 上的随机数生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
        
    # 设置 PyTorch 中的计算模式，确保生成的计算图具有可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))
                
def denormalize(tensor):
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    return tensor * std[:, None, None] + mean[:, None, None] 

def normalize(tensor):
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    return (tensor - mean[:, None, None]) / std[:, None, None]

def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()