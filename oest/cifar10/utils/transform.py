import os
import torch
from scipy import signal
import torch.nn.functional as F
import torch.nn as nn
import torchvision as tv
from torch.utils.data import *
import numpy as np
import random
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
from oest.cifar10.utils import normalize, denormalize


def cutout(images, mask_size=12):
    imgs = images.clone()
    h, w = imgs.shape[2], imgs.shape[3]

    cxmin, cxmax = 0, w - mask_size
    cymin, cymax = 0, h - mask_size

    for i in range(int(len(imgs)/2)):
        cx = random.randint(cxmin, cxmax)
        cy = random.randint(cymin, cymax)
        random_value = torch.randn(1).item()  
        imgs[i, :, cx:cx+mask_size, cy:cy+mask_size] = random_value
    
    imgs = denormalize(imgs)
    for i in range(int(len(imgs)/2)):
        cx = random.randint(cxmin, cxmax)
        cy = random.randint(cymin, cymax)
        imgs[int(len(imgs)/2)+i, :, cx:cx+mask_size, cy:cy+mask_size] = 0
    imgs = normalize(imgs)
    return imgs

def rotate(images):
    imgs = images.clone()
    for i in range(imgs.shape[0]):
        imgs[i] = torch.rot90(imgs[i], k=i%3+1, dims=[1, 2])
    return imgs

def noise(images):
    imgs = images.clone()
    return imgs + 0.1 * torch.randn_like(imgs)

def blur(images):
    imgs = images.clone()
    imgs = imgs.cpu().numpy()
    for i in range(imgs.shape[0]):
        for k in range(3):
            imgs[i, k, :, :] = gaussian_filter(imgs[i, k, :, :], 1)
    imgs = torch.from_numpy(imgs).to(images.device)
    return imgs

def perm(images):
    imgs = images.clone()
    x_mid, y_mid = int(imgs.shape[2] / 2) , int(imgs.shape[3] / 2)
    lu = imgs[:, :, :x_mid, :y_mid]
    ld = imgs[:, :, :x_mid, y_mid:]
    ru = imgs[:, :, x_mid:, :y_mid]
    rd = imgs[:, :, x_mid:, y_mid:]
    l = torch.cat([rd, ru], dim=3)
    r = torch.cat([ld, lu], dim=3)
    imgs = torch.cat([l, r], dim=2)
    return imgs

def sobel(images):
    imgs = images.clone()
    imgs[int(len(imgs)/2):] = denormalize(imgs[int(len(imgs)/2):])
    imgs = imgs.cpu().numpy()
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    imgs_sobel = imgs
    for i in range(imgs.shape[0]):
        imgs_sobel[i] = cv2.Sobel(imgs[i], -1, 1, 1, ksize=3)
    imgs = torch.from_numpy(np.transpose(imgs_sobel, (0, 3, 1, 2))).to(images.device)
    imgs[int(len(imgs)/2):] = normalize(imgs[int(len(imgs)/2):])
    # imgs = images.clone()
    # imgs = denormalize(imgs)
    # imgs = imgs.cpu().numpy()
    # imgs = np.transpose(imgs, (0, 2, 3, 1))
    # imgs_sobel = imgs
    # for i in range(imgs.shape[0]):
    #     imgs_sobel[i] = cv2.Sobel(imgs[i], -1, 1, 1, ksize=3)
    # imgs = torch.from_numpy(np.transpose(imgs_sobel, (0, 3, 1, 2))).to(images.device)
    # imgs = normalize(imgs)
    return imgs

# 尝试一些更hard的数据增强
def cutmix(images, alpha = 0.5):
    batch_size, channels, image_h, image_w = images.size()
    
    min_lam = 0.2  # 最小 lam 值
    max_lam = 0.8  # 最大 lam 值

    # 生成随机混合参数 lam
    while True:
        lam = np.random.beta(alpha, alpha)
        if min_lam <= lam <= max_lam:
            break
    
    # 随机选择另一张图像
    rand_index = torch.randperm(batch_size)
    shuffled_images = images[rand_index]
    
    # 计算混合区域的坐标
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))
    
    # 将混合区域的部分复制到原始图像中
    images[:, :, y0:y1, x0:x1] = shuffled_images[:, :, y0:y1, x0:x1]
    
    return images

# mixup
def mixup(images, alpha = 0.5):

    # 随机生成 MixUp 参数 lambda
    min_lam = 0.2  # 最小 lam 值
    max_lam = 0.8  # 最大 lam 值
    while True:
        lam = np.random.beta(alpha, alpha)
        if min_lam <= lam <= max_lam:
            break
    
    # 随机打乱样本顺序
    batch_size, channels, image_h, image_w = images.size()
    indices = torch.randperm(batch_size)
    shuffled_images = images[indices]

    # 使用 MixUp 参数 lambda 对图像进行混合
    mixed_images = lam * images + (1 - lam) * shuffled_images

    return mixed_images


