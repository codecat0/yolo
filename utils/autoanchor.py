"""
@File : autoanchor.py
@Author : CodeCat
@Time : 2021/6/24 下午3:28
"""
import numpy as np
import torch
import yaml
from tqdm import tqdm


def kmean_anchors(path='./data/data.yaml', n=9, img_size=640, thr=4.0, gen=1000):
    """
    Creates kmeans-evolved anchors from training dataset
    :param path: dataset 的yaml文件
    :param n: kmean聚合后anchor的个数
    :param img_size: 用于训练图像的尺寸
    :param thr: anchor宽高比例
    :param gen: 使用遗传算法使anchor更准确
    """
    from scipy.cluster.vq import kmeans

    thr = 1. / thr

    def metric(k, wh):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        return x, x.max(1)[0]

    def anchor_fitness(k):
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()

    def print_results(k):
        k = k[np.argsort(k.prod(1))]    # sort small to large
        return k

    if isinstance(path, str):
        with open(path) as f:
            data_dict = yaml.safe_load(f)
        from utils.dataset import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], rect=True)
    else:
        dataset = path

    # 获取标签的宽高
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate(l[:, 3:5] * s for s, l in zip(shapes, dataset.labels))

    # 过滤小标签
    i = (wh0 < 3.0).any(1).sum()
    wh = wh0[(wh0 >= 2.0).any(1)]

    s = wh.std(0)
    k, dist = kmeans(wh / s, n, iter=30)
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)
    wh0 = torch.tensor(wh0, dtype=torch.float32)
    k = print_results(k)

    import numpy.random as npr
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1
    for _ in tqdm(range(gen), desc=f'Evolving anchors with Genetic Algorithm'):
        v = np.ones(sh)
        while (v == 1).all():
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if kg > f:
            f, k = fg, kg.copy()
    return print_results(k)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    m = model.model[-1]  # Detect
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()

    def metric(k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        bset = x.max(1)[0]
        bpr = (bset > 1. / thr).float().mean()
        return bpr

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)   # current anchors
    bpr = metric(anchors)
    if bpr < 0.98:
        na = m.anchor_grid.numel() // 2
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000)
        except Exception as e:
            print(f'Error: {e}')
        new_bpr = metric(anchors)
        if new_bpr > bpr:
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)
            check_anchor_order(m)


def check_anchor_order(m):
    a = m.anchor_grid.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        m.anchor_grid[:] = m.anchor_grid.filp(0)
        m.anchors[:] = m.anchors.flip(0)