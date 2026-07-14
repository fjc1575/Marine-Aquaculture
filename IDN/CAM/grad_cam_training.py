import numpy as np
import torch

import cv2

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                        for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)

            scaled = cam

            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None, pre_mask=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

        self.pre_mask = pre_mask

    def minMax(self, tensor):
        maxs = tensor.max(dim=1, keepdim=True)[0]
        mins = tensor.min(dim=1, keepdim=True)[0]
        return (tensor - mins) / (maxs - mins)

    def combination(self, scores, grads_tensor):

        grads = self.minMax(grads_tensor)

        scores = self.minMax(scores)

        weights = torch.exp(scores) * grads - 0.71

        return weights.cpu().detach().numpy()

    def compute_cosine_similarity_scores(self, activation_cpu, scatter_channels, feature_map_size,
                                         selected_scatter=(0,)):
        """
        计算特征图与多个散射通道组合的余弦相似度得分。

        参数:
            activation_cpu (np.ndarray): 特征图，形状为 [C, H_fm, W_fm]。
            scatter_channels (np.ndarray): 散射通道，形状为 [3, H_input, W_input]。
            feature_map_size (tuple): 特征图的空间尺寸 (H_fm, W_fm)。
            selected_scatter (tuple): 要返回的散射通道索引的元组。

        返回:
            np.ndarray: 每个特征通道与组合散射通道的余弦相似度得分，形状为 [C]。
        """
        # 将 scatter_channels 转换为 numpy 数组
        scatter_channels = scatter_channels.cpu().detach().numpy()

        C, H_fm, W_fm = activation_cpu.shape

        num_scatter = scatter_channels.shape[0]

        correlation_scores = np.zeros(C, dtype=np.float32)

        # 检查 selected_scatter 是否为有效的元组
        if not isinstance(selected_scatter, (tuple, list)):
            raise ValueError("selected_scatter 必须是一个元组或列表")

        for scatter_idx in selected_scatter:
            if scatter_idx < 0 or scatter_idx >= num_scatter:
                raise ValueError(f"selected_scatter 包含无效的散射通道索引 {scatter_idx}，必须是 0, 1 或 2")

        # 下采样选定的散射通道到特征图尺寸并组合
        resized_scatter_combination = np.zeros((H_fm, W_fm), dtype=np.float32)

        for scatter_idx in selected_scatter:
            scatter_map = scatter_channels[scatter_idx, :, :]
            resized_scatter = cv2.resize(scatter_map, (W_fm, H_fm), interpolation=cv2.INTER_NEAREST)
            resized_scatter_combination += resized_scatter  # 组合方式可根据需求调整

        # 标准化特征图和散射通道
        resized_scatter_combination_flat = resized_scatter_combination.flatten().reshape(1, -1)

        feature_maps_flat = activation_cpu.reshape(C, -1)

        # 计算每个特征图与目标之间的余弦相似度
        cos_similarities = cosine_similarity(feature_maps_flat, resized_scatter_combination_flat.reshape(1, -1))

        # 取绝对值并存储
        correlation_scores = np.abs(cos_similarities).flatten()

        return correlation_scores

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):

        grads = np.mean(grads, axis=(2, 3))

        input_tensor = torch.from_numpy(self.pre_mask).cuda() * input_tensor

        correlation_scores = self.compute_cosine_similarity_scores(activations[0],
                                                                   input_tensor[0],
                                                                   (activations.shape[2],
                                                                    activations.shape[3]), selected_scatter=(0,))

        weights = self.combination(torch.from_numpy(correlation_scores).unsqueeze(0), torch.from_numpy(grads))

        return weights


