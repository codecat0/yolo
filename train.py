"""
@File : train.py
@Author : CodeCat
@Time : 2021/6/24 上午9:32
"""
import argparse
import math
import os
import random
import time
import logging
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.cuda import amp
import yaml
from torch.utils.tensorboard import SummaryWriter

from models.yolo import Model, Detect
from utils.dataset import LoadImagesAndLabels
from utils.loss import ComputeLoss
from utils.general import init_seeds, one_cycle, check_img_size, check_file
from utils.torch_utils import ModelEMA, warmup_lr_scheduler
from utils.autoanchor import check_anchors

logger = logging.getLogger(__name__)


def train(hyp, opt, tb_writer=None):
    save_dir, epochs, batch_size, weights = Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights

    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)     # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    result_file = wdir / 'results.txt'

    # 保存训练时的设置
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    device = opt.device
    cuda = device != 'cpu'
    init_seeds(1)
    # data dict
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)
    # 类别个数
    nc = int(data_dict['nc'])
    # 类别列表
    names = data_dict['names']
    # 训练集路径
    train_path = data_dict['train']
    # 验证集路径
    test_path = data_dict['val']

    # Model
    pretrained = weights.endswith('.pt') and opt.pre_train
    # pretrained = False
    if pretrained:
        ckpt = torch.load(weights, map_location=device)
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    # Freeze
    if opt.freeze_layers:
        # 冻结predictor和Detect之外的层
        for idx, module in enumerate(model.model):
            if isinstance(module, Detect):
                output_layer_indices = module.f
                output_layer_indices.append(idx)
        freeze_layer_indices = [x for x in range(len(model.model)) if x not in output_layer_indices]
        for idx in freeze_layer_indices:
            for parameter in model.model[idx].parameters():
                parameter.requires_grad_(False)
    else:
        # freeze_layers为False 冻结DarkNet53部分
        darkent_end_kayer_indice = 10
        for idx in range(darkent_end_kayer_indice+1):
            for parameter in model.model[idx].parameters():
                parameter.requires_grad_(False)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)    # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs    # scale weight_decay

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)    # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)    # apply decay

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    del pg0, pg1, pg2

    if opt.linear_lr:
        lf = lambda x: (1 - x/(epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt['ema']:
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            result_file.write_text(ckpt['training_results'])

        # Epoch
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    # Image sizes
    gs = max(int(model.stride.max()), 32)   # max stride
    nl = model.model[-1].nl     # number of detection layers
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    # Dataset
    train_dataset = LoadImagesAndLabels(path=train_path, img_size=imgsz, batch_size=batch_size,
                                        hyp=hyp, rect=opt.rect, cache_images=opt.cache_images,
                                        stride=gs)
    val_dataset = LoadImagesAndLabels(path=test_path, img_size=imgsz_test, batch_size=batch_size,
                                      hyp=hyp, rect=True, cache_images=opt.cache_images,
                                      stride=gs)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=nw,
        shuffle=not opt.rect,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    # max label class
    mlc = np.concatenate(train_dataset.labels, 0)[:, 0].max()
    # number of batches
    nb = len(train_dataloader)

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    if not opt.resume:
        labels = np.concatenate(train_dataset.labels, 0)
        c = torch.tensor(labels[:, 0])
        if tb_writer:
            tb_writer.add_histogram('classes', c, 0)

        # Anchors:
        if not opt.noautoanchor:
            check_anchors(train_dataset, model, thr=hyp['anchor_t'], imgsz=imgsz)
        model.half().float()

    # Model parameters
    hyp['box'] *= 3. / nl   # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl    # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0  # iou loss ratio

    # start training
    t0 = time.time()
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(4, device=device)

        pbar = enumerate(train_dataloader)
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        pbar = tqdm(pbar, total=nb)
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Warmup
            warmup_scheduler = None
            # if epoch == 0:
            #     warmup_factor = 1.0 / 1000
            #     warmup_iters = min(1000, nb - 1)
            #
            #     warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
            #     accumulate = 1

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgsz.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgsz.shape[2:]]
                    imgs = F.interpolate(imgs, size=sz, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            if ni % accumulate == 0 and warmup_scheduler is not None:
                warmup_scheduler.step()

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)
            if tb_writer:
                tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        # Log
        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',
                'learning_rate/lr0', 'learining_rate/lr1', 'learning_rate/lr2']

        for x, tag in zip(list(mloss[:-1]) + lr, tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)

        # Save model
        ckpt = {
            'epoch': epoch,
            'model': deepcopy(model).half(),
            'ema': deepcopy(ema.ema).half(),
            'updates': ema.updates,
            'optimizer': optimizer.state_dict()
        }
        torch.save(ckpt, last)
        del ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov3.pt', help='initial weights')
    parser.add_argument('--cfg', type=str, default='models/yolov3.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/data.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='[train, test] image size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--freeze_layers', type=bool, default=False, help='Freeze non-output layers')
    parser.add_argument('--pre_train', type=bool, default=True, help='pre train')
    opt = parser.parse_args()

    opt.data = check_file(opt.data)
    opt.cfg = check_file(opt.cfg)
    opt.hyp = check_file(opt.hyp)

    opt.save_dir = str(Path(opt.project))

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)

    tb_writer = SummaryWriter(opt.save_dir)
    train(hyp, opt, tb_writer)