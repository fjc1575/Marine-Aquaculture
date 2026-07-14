import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss


from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
# Recommend


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        # 计算 Dice 分子部分（交集）
        intersection = torch.sum(input * target)

        # 计算 Dice 分母部分（预测区域面积 + 真实区域面积）
        union = torch.sum(input) + torch.sum(target)

        # 计算 Dice 分数
        dice = (2.0 * intersection) / (union + 1e-6)  # 添加一个小的常数以避免除以零

        # 计算 Dice Loss（1 - Dice 分数，使其成为损失）
        loss = 1 - dice

        return loss

class FocalLoss_Edge(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, ignore_index=-100):
        super(FocalLoss_Edge, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (N, 2, H, W)
        # targets: (N, H, W)

        # 计算交叉熵损失
        logpt = -F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        # 计算Focal Loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        # 计算交叉熵损失
        loss = F.cross_entropy(input, target, weight=self.weight, reduction='mean')
        return loss

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss
# 定义IoU损失
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # pred = F.sigmoid(pred)

        # flatten label and prediction tensors
        pred = pred.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (pred * targets).sum()
        total = (pred + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU

class BceIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceIoULoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.iou = IoULoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target.float())
        iouloss = self.iou(pred, target.float())

        loss = iouloss + bceloss

        return loss


class ScatteringPenaltyLoss(nn.Module):
    def __init__(self, use_soft_constraint=True):
        super(ScatteringPenaltyLoss, self).__init__()
        self.use_soft_constraint = use_soft_constraint

    def calculate_thresholds_batch(self, freeman_images, binary_masks):
        """
        计算批次数据中每张 Freeman 伪彩色图像的三个通道在目标区域内的上下阈值。

        参数:
        - freeman_images: 形状为 (batch_size, 3, height, width) 的张量，表示 Freeman 分解的伪彩色图像。
        - binary_masks: 形状为 (batch_size, height, width) 的张量，值为1表示目标区域，值为0表示背景。

        返回:
        - thresholds: 形状为 (batch_size, 3, 2) 的张量，包含每张图像每个通道的上下阈值。
        """
        freeman_images = freeman_images.permute(0, 2, 3, 1)  # 转换维度
        batch_size, height, width, channels = freeman_images.shape

        thresholds = []
        l = 5
        u = 100 - l
        for i in range(batch_size):
            binary_mask = binary_masks[i].unsqueeze(-1).expand(-1, -1, channels)
            binary_mask_bool = binary_mask.bool()

            masked_image = freeman_images[i][binary_mask_bool].view(-1, channels)


            # 计算每个通道的上下阈值
            lower_percentiles = torch.quantile(masked_image, l / 100.0, dim=0)
            upper_percentiles = torch.quantile(masked_image, u / 100.0, dim=0)

            # 合并成阈值矩阵 (3, 2)
            threshold = torch.stack((lower_percentiles, upper_percentiles), dim=-1)
            thresholds.append(threshold)

        # 拼接阈值矩阵，形成 (batch_size, 3, 2) 的张量
        thresholds = torch.stack(thresholds, dim=0)

        return thresholds

    def forward(self, predicted_classes, image, wishart_label):
        # 计算每张图像的阈值
        thresholds = self.calculate_thresholds_batch(image, wishart_label)

        batch_size = image.size(0)

        # 初始化一个列表来存储每张图片的惩罚
        penalties = []

        for i in range(batch_size):
            # 提取当前图片的阈值
            lower_Ds, upper_Ds = thresholds[i, 0]
            lower_Ss, upper_Ss = thresholds[i, 1]
            lower_Vs, upper_Vs = thresholds[i, 2]

            # 提取当前图片的通道
            Ds = image[i, 0, :, :]  # 红色通道（双回波散射）
            Vs = image[i, 1, :, :]  # 绿色通道（体积散射）
            Ss = image[i, 2, :, :]  # 蓝色通道（表面散射）

            # 提取当前图片的二值掩码
            mask = (predicted_classes[i] == 1)

            # 选择当前图片的像素值
            selected_Ds = Ds[mask]
            selected_Vs = Vs[mask]
            selected_Ss = Ss[mask]

            if self.use_soft_constraint:
                # 软约束使用 sigmoid 函数
                penalty_ds = torch.sigmoid(selected_Ds - lower_Ds) * torch.sigmoid(upper_Ds - selected_Ds)
                penalty_ss = torch.sigmoid(selected_Ss - lower_Ss) * torch.sigmoid(upper_Ss - selected_Ss)
                penalty_vs = torch.sigmoid(selected_Vs - lower_Vs) * torch.sigmoid(upper_Vs - selected_Vs)
            else:
                # 严格约束使用 clamping
                penalty_ds = torch.clamp(selected_Ds - lower_Ds, min=0) + torch.clamp(upper_Ds - selected_Ds, min=0)
                penalty_ss = torch.clamp(selected_Ss - lower_Ss, min=0) + torch.clamp(upper_Ss - selected_Ss, min=0)
                penalty_vs = torch.clamp(selected_Vs - lower_Vs, min=0) + torch.clamp(upper_Vs - selected_Vs, min=0)

            # 合并当前图片的所有区域的惩罚
            penalty = penalty_ds + penalty_ss + penalty_vs

            # 按选定像素的数量归一化惩罚
            penalty_mean = penalty.mean() if penalty.numel() > 0 else torch.tensor(0.0, device=image.device)
            penalties.append(penalty_mean)

        # 返回所有图片的平均惩罚
        return torch.stack(penalties).mean()



if __name__ == '__main__':

    # 初始化带有软约束的散射惩罚损失
    penalty_loss_soft = ScatteringPenaltyLoss(
        use_soft_constraint=True
    )

    # 初始化带有严格约束的散射惩罚损失
    penalty_loss_strict = ScatteringPenaltyLoss(
        use_soft_constraint=False
    )

    # 生成随机测试数据
    batch_size = 2
    height = 512
    width = 512

    # 模拟一批预测类别（类别索引为 0 或 1）
    predicted_classes = torch.randint(0, 2, (batch_size, height, width))

    # 模拟一批输入图像（3 个通道分别对应 Ds、Vs、Ss）
    image = torch.randn(batch_size, 3, height, width)

    wishart_label = torch.randint(0, 2, (batch_size, height, width))

    # 使用软约束计算惩罚
    penalty_soft = penalty_loss_soft(predicted_classes, image, wishart_label)
    print("Soft Constraint Penalty:", penalty_soft.item())

    # 使用严格约束计算惩罚
    penalty_strict = penalty_loss_strict(predicted_classes, image, wishart_label)
    print("Strict Constraint Penalty:", penalty_strict.item())
