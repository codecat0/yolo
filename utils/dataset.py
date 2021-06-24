"""
@File : dataset.py
@Author : CodeCat
@Time : 2021/6/18 下午7:19
"""
import glob
import hashlib
import os
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywhn2xyxy

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
                 img_size=640,   # 预处理后输出的图像尺寸
                 batch_size=16,
                 hyp=None,  # 超参数字典
                 rect=False,    # 是否使用rectangular training
                 cache_images=False,    # 是否缓存图片到内存
                 stride=32,    # 将图片尺寸调整到stride的整数倍
                 pad=0.0):
        self.img_size = img_size
        self.hyp = hyp
        self.rect = rect
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

        # 查看缓存信息
        nf, nm, ne, nc, n = cache.pop('results')
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=d, total=n, initial=n)

        # 读取缓存信息
        # cache = {'im_file1': [label, shape], 'im_file2': [label, shape], ..., 'hash': hash_value, 'results': (nf, nm, ne, nc, n)}
        cache.pop('hash')   # 移除hash
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        # 更新图像文件信息
        self.img_files = list(cache.keys())
        # 更新图像标签信息
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        # 图像的数量
        n = len(shapes)
        # batch索引
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        # batch的数量
        nb = bi[-1] + 1
        self.batch = bi
        self.n = n
        self.indices = range(n)

        # 是否使用类似原图像比例的矩形进行训练，即最长边为img_size，而不是img_size x img_size
        if self.rect:
            s = self.shapes
            # 获取每个图像的高/宽比
            ar = s[:, 1] / s[:, 0]
            # 按照高宽比进行排序，使得每个batch中的数据具有类似的高宽比
            irect = ar.argsort()
            # 根据排序后的结果重新设置图像顺序，标签顺序和图像shape数据
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]

            # 计算每一个batch中的数据的图像尺寸
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                # 获取第i个batch中的最大高宽比和最小高宽比
                mini, maxi = ari.min(), ari.max()
                # 如果最大的高宽比<1, 说明h<w，将w设置为img_size
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                # 如果最小高宽比>1，说明h>w，将h设置为img_size
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            # 计算每个batch输入网络的shape值(向上取整为stride(32)的整数倍)
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            # 用于记录缓存图像占用RAM大小
            gb = 0
            # 原始图像尺寸，resize后的图像尺寸
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            # for i in range(n):
            #       load_image(self, i)
            # 采用8线程
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                gb += self.imgs[i].nbytes
                pbar.desc = f'Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

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

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        hyp = self.hyp
        # 导入图像
        img, (h0, w0), (h, w) = load_image(self, index)

        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
        img, ratio, pad = letterbox(img, shape, auto=False)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP resacling

        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        nL = len(labels)    # 标签的数量
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0]   # 将高度标准化 0-1
            labels[:, [1, 3]] /= img.shape[1]   # 将宽度标准化 0-1

        # (batch中图像索引，框中物体类别，x, y, w, h)
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert BGR to RGB and HWC to CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            # 添加batch中图像索引
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def load_image(self, index):
    """
    从数据集中加载一张图像，返回图像，原始图像尺寸，resize后的尺寸
    """
    img = self.imgs[index]
    # 没有缓存图像信息
    if img is None:
        path = self.img_files[index]
        # BGR
        img = cv2.imread(path)
        assert img is not None, 'Image Not Found' + path
        # 原始图像尺寸
        h0, w0 = img.shape[:2]
        # img_size 设置的是预处理后输出的图像尺寸
        r = self.img_size / max(h0, w0)
        if r != 1:
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]


def letterbox(img,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scaleFill=False,
              scaleup=True,
              stride=32):
    """将图像调整到指定大小"""

    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 对于大于指定输入大小的图像进行调整，小于的不变（for better test mAP）
    if not scaleup:
        r = min(r, 1.0)


    # 计算padding
    ratio = r, r
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dh, dw = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]   # wh padding
    # 保证原图比例不变
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    # 直接将图像缩放到指定尺寸
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    # 将padding分到左右，上下两侧
    dw /= 2
    dh /= 2

    if shape != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)