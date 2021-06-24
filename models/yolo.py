"""
@File : yolo.py
@Author : CodeCat
@Time : 2021/6/18 上午11:04
"""
import argparse
import yaml
from pathlib import Path
from copy import deepcopy
import math

import torch
import torch.nn as nn


from models.common import Conv, Bottleneck, Concat
from utils.general import make_divisible
from utils.torch_utils import initialize_weights
from utils.autoanchor import check_anchor_order


class Detect(nn.Module):
    stride = None

    def __init__(self, nc=80, anchors=(), ch=()):
        super(Detect, self).__init__()
        self.nc = nc    # 类别的数量
        self.no = nc + 5    # 每个anchor输出维度的数量
        self.nl = len(anchors)  # 检测特征层的数量
        self.na = len(anchors[0]) // 2  # anchor的数量
        self.grid = [torch.zeros(1)] * self.nl  # 网格初始化
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))    # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # 预测输出
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape  # (bs, 255, 20, 20)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # (bs, 3, 20, 20, 85)

            # 预测
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                # y = x[i].sigmoid()
                # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                y = x[i].clone()
                y[..., 0:2] = (torch.sigmoid(y[..., 0:2]) + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = torch.exp(y[...:2:4]) * self.anchor_grid[i]   # wh
                torch.sigmoid(y[..., 4:])
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()


class Model(nn.Module):
    def __init__(self, cfg='yolo3.yaml', ch=3, nc=None, anchors=None):  # 模型参数，输入图像通道数，类别数
        super(Model, self).__init__()
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.yaml = yaml.safe_load(f)

        # 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc
        if anchors:
            self.yaml['anchors'] = anchors
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])

        # Detect()
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            x = m(x)
            y.append(x if m.i in self.save else None)

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def parse_model(d, ch):   # 模型参数字典，输入通道数
    anchors = d['anchors']
    nc = d['nc']

    na = len(anchors[0]) // 2   # anchor的数量
    no = na * (nc + 5)  # 输出的通道数=anchor的数量 x（类别数+5）

    layers, save, c2 = [], [], []

    # from, number, module, args
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass

        if m in [Conv, Bottleneck]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2, 8)

            args = [c1, c2, *args[1:]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])

        # 创建module
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)

        # 添加index 和 ‘from’的index
        m_.i, m_.f = i, f

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3.yaml', help='model.yaml')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    # Create model
    model = Model(opt.cfg).to(opt.device)

    for idx, module in enumerate(model.model):
        if isinstance(module, Detect):
            output_layer_indices = module.f
            output_layer_indices.append(idx)

    print(output_layer_indices)
