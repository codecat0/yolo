"""
@File : general.py
@Author : CodeCat
@Time : 2021/6/18 下午5:37
"""
import math
import torch
import numpy as np


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw    # xmin
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh    # ymin
    y[:, 2] = w * (x[:, 2] + x[:, 2] / 2) + padw    # xmax
    y[:, 3] = h * (x[:, 3] + x[:, 3] / 2) + padh    # ymax
    return y


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2   # center x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2   # center y
    y[:, 2] = x[:, 2] - x[:, 0]     # width
    y[:, 3] = x[:, 3] - x[:, 1]     # height
    return y