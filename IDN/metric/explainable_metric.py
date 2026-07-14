import torch
import torchvision
import math
from pytorch_grad_cam import GradCAM, XGradCAM

from architectures.BPCAMNET import BPCAMNET
from parameters import setupArgs
from train.train_bpcamnet import Supervision_Train
from utils.cuda import setupCuda
from utils.seed import setSeed


def blur_image(input_image):
    """
    blur the input image tensor
    """
    return torchvision.transforms.functional.gaussian_blur(input_image, kernel_size=[11, 11], sigma=[5,5])

def grey_image(input_image):
    """
    generate a grey image tensor with same shape as input
    """
    # create an uniform grey image with same size as input_image
    grey = 0.5 * torch.ones_like(input_image).to(input_image.device)
    return grey

def black_image(input_image):
    """
    generate a grey image tensor with same shape as input
    """
    # create an uniform grey image with same size as input_image
    black = torch.zeros_like(input_image).to(input_image.device)
    return black

import torch
import torchvision
import math

def compute_iou(pred, target):
    """
    计算 IoU 指标
    :param pred: 模型预测的分割结果（二值化）
    :param target: 真实分割掩码（二值化）
    :return: IoU 值
    """
    intersection = (pred & target).sum()  # 交集
    union = (pred | target).sum()  # 并集
    return intersection / union if union > 0 else torch.tensor(0.0)

class RelevanceMetricIoU:
    """
    基于 IoU 的插入和删除评估方法
    """
    def __init__(self, model, n_steps, batch_size, baseline="blur"):
        """
        初始化
        :param model: 分割模型
        :param n_steps: 插入或删除的步数
        :param batch_size: 批量大小
        :param baseline: 基线类型 ("blur" 或 "grey")
        """
        self.model = model
        self.sigmoid = torch.nn.Sigmoid()
        self.n_steps = n_steps
        self.batch_size = batch_size

        if baseline == "blur":
            self.baseline_fn = blur_image
        elif baseline == "grey":
            self.baseline_fn = grey_image
        else:
            self.baseline_fn = black_image

    def __call__(self, image, saliency_map, target_mask):
        """
        计算显著性图的 IoU 曲线
        :param image: 输入图像
        :param saliency_map: 显著性图
        :param target_mask: 真实分割掩码
        :return: AUC 和 IoU 曲线
        """
        h, w = image.shape[-2:]
        # 生成基线图像
        baseline = self.baseline_fn(image)
        # 像素排序
        sorted_index = torch.flip(saliency_map.view(-1, h * w).argsort(), dims=[-1])
        # 生成插入或删除的样本


        samples = self.generate_samples(sorted_index, image, baseline)

        iou_scores = torch.zeros(self.n_steps).to(image.device)  # 存储每一步的 IoU

        for idx in range(math.ceil(samples.shape[0] / self.batch_size)):
            selection_slice = slice(idx * self.batch_size, min((idx + 1) * self.batch_size, samples.shape[0]))
            batch_samples = samples[selection_slice]
            with torch.no_grad():
                # output, attn = self.model(batch_samples)
                output = self.model(batch_samples)
                pred = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
                for i, prediction in enumerate(pred):
                    iou_scores[idx * self.batch_size + i] = compute_iou(prediction, target_mask)

        auc = torch.sum(iou_scores) / self.n_steps  # 计算 AUC
        return auc, iou_scores

    def generate_samples(self, *args, **kwargs):
        raise NotImplementedError

class InsertionIoU(RelevanceMetricIoU):
    """
    基于 IoU 的插入评估
    """
    def generate_samples(self, index, image, baseline):
        h, w = image.shape[-2:]
        pixels_per_steps = (h * w + self.n_steps - 1) // self.n_steps
        samples = torch.ones(self.n_steps, *image.shape[-3:]).to(image.device)
        samples = samples * baseline
        for step in range(self.n_steps):
            pixels = index[:, :pixels_per_steps * (step + 1)]
            samples[step].view(-1, h * w)[..., pixels] = image.view(-1, h * w)[..., pixels]
        return samples

class DeletionIoU(RelevanceMetricIoU):
    """
    基于 IoU 的删除评估
    """
    def generate_samples(self, index, image, baseline):
        h, w = image.shape[-2:]
        pixels_per_steps = (h * w + self.n_steps - 1) // self.n_steps
        samples = torch.ones(self.n_steps, *image.shape[-3:]).to(image.device)
        samples = samples * image
        for step in range(self.n_steps):
            pixels = index[:, :pixels_per_steps * (step + 1)]
            samples[step].view(-1, h * w)[..., pixels] = baseline.view(-1, h * w)[..., pixels]
        return samples


if __name__ == '__main__':
    from PIL import Image
    from torchvision.utils import save_image
    import numpy as np

    opt = setupArgs()
    setSeed(opt)
    setupCuda(opt)

    model = Supervision_Train.load_from_checkpoint(r'C:\polsar\code\PolSARSeg\SGSCAM\bpcamnet-2222-gf3.ckpt', opt=opt)
    image = Image.open(r'C:\polsar\code\PolSARSeg\asset\gf3\1 (20).png')
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0).to('cuda')

    model.eval()

    output = model(image)
    saliency_map = torch.rand(1,512, 512).to('cuda')  # 示例显著性图
    target_mask = torch.randint(0, 2, (1, 512, 512)).to('cuda')  # 示例目标掩码

    # 初始化插入和删除评估
    insertion = InsertionIoU(model=model, n_steps=20, batch_size=10, baseline="black")
    deletion = DeletionIoU(model=model, n_steps=20, batch_size=10, baseline="black")

    # 计算指标
    insertion_auc, insertion_scores = insertion(image, saliency_map, target_mask)
    deletion_auc, deletion_scores = deletion(image, saliency_map, target_mask)

    print(f"Insertion AUC: {insertion_auc}")
    print(f"Deletion AUC: {deletion_auc}")

