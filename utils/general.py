"""
@File : general.py
@Author : CodeCat
@Time : 2021/6/18 下午5:37
"""

import math
import random
import glob
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw    # xmin
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh    # ymin
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw    # xmax
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh    # ymax
    return y


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2   # center x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2   # center y
    y[:, 2] = x[:, 2] - x[:, 0]     # width
    y[:, 3] = x[:, 3] - x[:, 1]     # height
    return y


def wh_iou(wh1, wh2):
    wh1 = wh1[:, None]  # [N, 1, 2]
    wh2 = wh2[None]     # [1, M, 2]
    inter = torch.min(wh1, wh2).prod(2)     # [N, M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIou=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[4]
    else:   # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIou:
        # 获取一个最小闭包，可以将box1 和 box2包含在内
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIou or DIoU:
            # 获取最小闭包的对角线距离
            c2 = cw ** 2 + ch ** 2 + eps
            # 获取box1 和 box2的中心点之间的距离
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                # DIoU
                return iou - rho2 / c2
            elif CIou:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 + eps - iou + v)
                # CIoU
                return iou - rho2 / c2 - v * alpha
        else:
            c_area = cw * ch + eps
            # GIoU
            return iou - (c_area - union) / c_area
    else:
        return iou


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_file(file):
    file = str(file)
    if Path(file).is_file() or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)
        assert len(files), f'File not fount: {file}'
        return file[0]