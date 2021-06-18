"""
@File : dataset.py
@Author : CodeCat
@Time : 2021/6/18 下午7:19
"""
import glob
import hashlib
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

# 获取一组文件路径列表的hash值
def get_hash(paths):
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    # 将文件大小进行hash处理
    h = hashlib.md5(str(size).encode())
    # 将文件名进行hash处理
    h.update(''.join(paths).encode())
    return h.hexdigest()


# 找到图像exif信息中对应旋转信息的key值
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


# 获取图像的原始尺寸
def exif_size(img):
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        # 顺时针旋转90度h或者逆时针旋转90度
        if rotation == 6 or rotation == 8:
            s = (s[1], s[0])
    except:
        pass

    return s


class LoadImagesAndLabels(Dataset):   # for training/testing
    def __init__(self,
                 path,  # 指向data/images的路径
                 img_size=416,   # 预处理后输出的图像尺寸
                 batch_size=16,
                 augment=False,  # 图像增强 （颜色空间）
                 hyp=None,  # 超参数字典
                 rect=False,    # 是否使用rectangular training
                 cache_images=False,    # 是否缓存图片到内存
                 stride=32,    # 将图片尺寸调整到stride的整数倍
                 pad=0.0):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.mosaic = self.augment and not self.rect    # 将4张图片拼接在一张马赛克图像中
        self.path = path

        # 定义images
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '*.*'), recursive=True)
                else:
                    raise Exception(f'{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, 'No images found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}\nSee {help_url}')

        # 定义labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        # 检查缓存文件是否存在
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        if cache_path.is_file():
            # 加载缓存，并将缓存存在信息设为Ture
            cache, exists = torch.load(cache_path), True
            # 检查缓存的hash值是否发生改变,当缓存文件的名称和大小未发生变化时，hash值不变
            if cache['hash'] != get_hash(self.label_files + self.img_files):
                # 重写缓存信息, 并将缓存存在信息设置为False
                cache, exists = self.cache_labels(cache_path), False
        else:
            cache, exists = self.cache_labels(cache_path), False

        # 读取cache


    # 将数据集的labels信息加载到缓存，检查images信息和读取image形状
    def cache_labels(self, path=Path('./labels.cache')):
        x = {}
        # labels丢失数量，labels正常数量，labels为空的数量，labels出现重复的数量
        nm, nf, ne, nc = 0, 0, 0, 0
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # 检查图像信息
                im = Image.open(im_file)
                im.verify()
                shape = exif_size(im)

                # 检查labels信息
                if os.path.isfile(lb_file):
                    nf += 1
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape]
            except Exception as e:
                nc += 1
                print(f'WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()

        if nf == 0:
            print(f'WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i+1

        try:
            # 保存缓存文件
            torch.save(x, path)
            print(f'New cache created: {path}')
        except Exception as e:
            print(f'WARNING: Cache directory {path.parent} is not writeable: {e}')

        return x
