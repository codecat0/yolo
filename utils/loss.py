"""
@File  : loss.py
@Author: CodeCat
@Time  : 2021/6/19 23:07
"""
import torch
import torch.nn as nn

from utils.general import wh_iou, bbox_iou


class ComputeLoss:
    """计算损失"""
    def __init__(self, model):
        super(ComputeLoss, self).__init__()
        # 获取模型在哪个设备上运行
        device = next(model.parameters()).device

        # 超参数
        h = model.hyp

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # 标签平滑
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        # Focal loss
        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Detect() module
        det = model.model[-1]

        # cls损失，obj损失，iou loss ratio，超参数
        self.BCEcls, self.BCEobj, self.gr, self.hyp = BCEcls, BCEobj, model.gr, h

        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        # Losses
        for i, pi in enumerate(p):
            # image, anchor, gridy, gridx
            b, a, gj, gi = indices[i]
            # tobj: (b, na, w, h)
            tobj = torch.zeros_like(pi[..., 0], device=device)

            # targets的数目
            n = b.shape[0]
            if n:
                # 对应匹配到正样本的预测信息
                ps = pi[b, a, gj, gi]

                # Regression
                # pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pxy = ps[:, :2].sigmoid()
                pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
                # predicted box
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIou=True)
                lbox += (1.0 - iou).mean()  # ciou loss

                # Objectness
                # self.gr = 1.0
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)      # iou ratio

                # Classification
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

            lobj += self.BCEobj(pi[..., 4], tobj)

        # x各种损失的权重
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        # batch size
        bs = tobj.shape[0]
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        """Build targets for compute_loss(), input targets(image,class,x,y,w,h)"""
        # anchors的数量
        na = self.na
        # targets的数量
        nt = targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        # normalized to gridsapce gain
        gain = torch.ones(6, device=targets.device)

        for i in range(self.nl):
            anchors = self.anchors[i]
            # 获取特征图的尺寸
            # p : [(b, 3, w, h, 5+classes),..]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
            # [3] -> [3, 1] -> [3, nt]
            at = torch.arange(na).view(na, 1).repeat(1, nt)

            # Match targets to anchors
            a, t, offsets = [], targets * gain, 0
            if nt:
                # iou_t = 0.2
                # j: [3, nt]
                j = wh_iou(anchors, t[:, 4:6]) > self.hyp['iou_t']
                # t.repeat(na, 1, 1): [nt, 6] -> [3, nt, 6]
                # 获取iou值大于阈值的anchor与target对应信息
                a, t = at[j], t.repeat(na, 1, 1)[j]

            # Define
            # image, class
            b, c = t[:, :2].long().T
            # grid xy
            gxy = t[:, 2:4]
            # grid wh
            gwh = t[:, 4:6]
            # 匹配targets所在grid cell的左上角坐标
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            # Append
            # image, anchor, grid indcices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp(0, gain[2] - 1)))
            # gt box相对anchor的x, y偏移量以及w,h
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            # anchors
            anch.append(anchors[a])
            # class
            tcls.append(c)

        return tcls, tbox, indices, anch


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn    # 必须为 nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


