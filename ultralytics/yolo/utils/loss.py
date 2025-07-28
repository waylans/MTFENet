# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F


from .metrics import bbox_iou, probiou, bbox_mpdiou, bbox_inner_iou, bbox_focaler_iou, bbox_inner_mpdiou, bbox_focaler_mpdiou, wasserstein_loss, WiseIouLoss
from .tal import bbox2dist
import math
import torch.cuda.amp as amp


class FocalLossV1(nn.Module):
    """https://github.com/CoinCheung/pytorch-loss/blob/master/focal_loss.py"""
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')
        # self.crit = EMASlideLoss(nn.BCEWithLogitsLoss(reduction='none'))  # Exponential Moving Average Slide Loss
        # self.bce = SlideLoss(nn.BCEWithLogitsLoss(reduction='none')) # Slide Loss
        # self.bce = FocalLoss_YOLO(alpha=0.25, gamma=1.5) # FocalLoss
        # self.bce = VarifocalLoss_YOLO(alpha=0.75, gamma=2.0) # VarifocalLoss
        # self.bce = QualityfocalLoss_YOLO(beta=2.0) # QualityfocalLoss

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

# version 2: user derived grad computation
class FocalSigmoidLossFuncV2(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        #  logits = logits.float()

        probs = torch.sigmoid(logits)
        coeff = (label - probs).abs_().pow_(gamma).neg_()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        ce_term1 = log_probs.mul_(label).mul_(alpha)
        ce_term2 = log_1_probs.mul_(1. - label).mul_(1. - alpha)
        ce = ce_term1.add_(ce_term2)
        loss = ce * coeff

        ctx.vars = (coeff, probs, ce, label, gamma, alpha)

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        (coeff, probs, ce, label, gamma, alpha) = ctx.vars

        d_coeff = (label - probs).abs_().pow_(gamma - 1.).mul_(gamma)
        d_coeff.mul_(probs).mul_(1. - probs)
        d_coeff = torch.where(label < probs, d_coeff.neg(), d_coeff)
        term1 = d_coeff.mul_(ce)

        d_ce = label * alpha
        d_ce.sub_(probs.mul_((label * alpha).mul_(2).add_(1).sub_(label).sub_(alpha)))
        term2 = d_ce.mul(coeff)

        grads = term1.add_(term2)
        grads.mul_(grad_output)

        return grads, None, None, None


class FocalLossV2(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV2()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class SlideLoss(nn.Module):
    def __init__(self, loss_fcn):
        super(SlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element

    def forward(self, pred, true, auto_iou=0.5):
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class EMASlideLoss:
    def __init__(self, loss_fcn, decay=0.999, tau=2000):
        super(EMASlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        self.is_train = True
        self.updates = 0
        self.iou_mean = 1.0
    
    def __call__(self, pred, true, auto_iou=0.5):
        if self.is_train and auto_iou != -1:
            self.updates += 1
            d = self.decay(self.updates)
            self.iou_mean = d * self.iou_mean + (1 - d) * float(auto_iou.detach())
        auto_iou = self.iou_mean
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss

class VarifocalLoss_YOLO(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        """Initialize the VarifocalLoss class."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_score, gt_score):
        """Computes varfocal loss."""
        
        weight = self.alpha * (pred_score.sigmoid() - gt_score).abs().pow(self.gamma) * (gt_score <= 0.0).float() + gt_score * (gt_score > 0.0).float()
        with torch.cuda.amp.autocast(enabled=False):
            return F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') * weight

class QualityfocalLoss_YOLO(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred_score, gt_score, gt_target_pos_mask):
        # negatives are supervised by 0 quality score
        pred_sigmoid = pred_score.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_score.shape)
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(pred_score, zerolabel, reduction='none') * scale_factor.pow(self.beta)
        
        scale_factor = gt_score[gt_target_pos_mask] - pred_sigmoid[gt_target_pos_mask]
        with torch.cuda.amp.autocast(enabled=False):
            loss[gt_target_pos_mask] = F.binary_cross_entropy_with_logits(pred_score[gt_target_pos_mask], gt_score[gt_target_pos_mask], reduction='none') * scale_factor.abs().pow(self.beta)
        return loss

class FocalLoss_YOLO(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, gamma=1.5, alpha=0.25):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, label):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if self.alpha > 0:
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss
class tversky(nn.Module):
    def __init__(self, smooth=1):
        super(tversky, self).__init__()
        self.smooth = smooth



    def forward(self, logits, label,alpha=0.7):
        '''
        args: logits: tensor of shape (1, H, W)
        args: label: tensor of shape (1, H, W)
        '''
        probs = torch.sigmoid(logits)

        true_pos = torch.sum(label * probs)
        false_neg = torch.sum(label * (1 - probs))
        false_pos = torch.sum((1 - label) * probs)
        loss = (true_pos + self.smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + self.smooth)

        return 1-loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', weight=None, ignore_lb=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (1, H, W)
        args: label: tensor of shape (1, H, W)
        '''
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)


        ignore_mask = label == self.ignore_lb
        lb_one_hot = torch.zeros_like(probs)
        lb_one_hot[label == 1] = 1
        lb_one_hot[ignore_mask] = 0

        # Compute loss
        numer = torch.sum(probs * lb_one_hot)
        denom = torch.sum(probs + lb_one_hot)

        loss = 1 - (2 * numer + self.smooth) / (denom + self.smooth)

        return loss



class IoULoss(nn.Module):
    '''https://blog.csdn.net/lwf1881/article/details/123725202'''
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.nwd_loss = False
        self.iou_ratio = 0.5
        self.use_wiseiou = False
        
        if self.use_wiseiou:
            self.wiou_loss = WiseIouLoss(ltype='MPDIoU', monotonous=False, inner_iou=True, focaler_iou=False)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,mpdiou_hw=None):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if self.use_wiseiou:
            # wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95).unsqueeze(-1)
            # wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'scale':0.0}).unsqueeze(-1) # Wise-ShapeIoU,Wise-Inner-ShapeIoU,Wise-Focaler-ShapeIoU
            wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'mpdiou_hw':mpdiou_hw[fg_mask]}).unsqueeze(-1) # Wise-MPDIoU,Wise-Inner-MPDIoU,Wise-Focaler-MPDIoU
            loss_iou = (wiou * weight).sum() / target_scores_sum
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
             # iou = bbox_inner_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, ratio=0.7)
            # iou = bbox_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask])
            # iou = bbox_inner_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask], ratio=0.7)
            # iou = bbox_focaler_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, d=0.0, u=0.95)
            # iou = bbox_focaler_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask], d=0.0, u=0.95)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        
        if self.nwd_loss:
            nwd = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            nwd_loss = ((1.0 - nwd) * weight).sum() / target_scores_sum
            loss_iou = self.iou_ratio * loss_iou +  (1 - self.iou_ratio) * nwd_loss

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()
