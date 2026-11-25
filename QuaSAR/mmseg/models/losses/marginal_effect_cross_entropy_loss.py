import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class MarginalEffectCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss with Marginal Effect dynamic weighting.

    根据当前 Dataset 的高/中/低质量样本比例动态调整 loss 权重.

    Args:
        alpha (float): 高质量样本边际衰减系数. Default=1.0
        beta (float): 高质量样本指数衰减速率. Default=2.0
        gamma (float): 中质量样本峰值权重. Default=0.3
        m_star (float): 中质量样本收益峰值位置. Default=0.5
        delta (float): 中质量样本收益峰值宽度. Default=0.4
        l0 (float): 低质量样本鲁棒性权重. Default=0.2
        use_sigmoid (bool): 是否使用 sigmoid. Default=False
        reduction (str): 'none' | 'mean' | 'sum'. Default='mean'
        loss_weight (float): 损失权重. Default=1.0
    """

    def __init__(self,
                 alpha=1.0,
                 beta=2.0,
                 gamma=0.3,
                 m_star=0.5,
                 delta=0.4,
                 l0=0.2,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.m_star = m_star
        self.delta = delta
        self.l0 = l0

        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.register_buffer('h_ratio', torch.tensor(0.0))
        self.register_buffer('m_ratio', torch.tensor(0.0))
        self.register_buffer('l_ratio', torch.tensor(0.0))

    def set_data_ratios(self, high_ratio, mid_ratio, low_ratio):
        """在训练开始前设置高/中/低数据比例"""
        self.h_ratio.fill_(high_ratio)
        self.m_ratio.fill_(mid_ratio)
        self.l_ratio.fill_(low_ratio)

    def compute_weights(self):
        """根据比例计算权重"""
        high_w = self.alpha * torch.exp(-self.beta * self.h_ratio)
        mid_w = self.gamma * torch.exp(-((self.m_ratio - self.m_star) ** 2) / (2 * self.delta ** 2))
        low_w = self.l0 * torch.log1p(self.l_ratio * 10)
        return high_w + mid_w + low_w

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward with marginal effect weighting."""
        reduction = reduction_override if reduction_override else self.reduction

        # 计算基础 cross entropy
        if self.use_sigmoid:
            ce_loss = F.binary_cross_entropy_with_logits(
                cls_score, label.float(), reduction='none')
        else:
            ce_loss = F.cross_entropy(
                cls_score, label, reduction='none', ignore_index=255)

        # 加权
        marginal_weight = self.compute_weights()
        ce_loss = ce_loss * marginal_weight

        # 聚合
        loss = weight_reduce_loss(
            ce_loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return self.loss_weight * loss
