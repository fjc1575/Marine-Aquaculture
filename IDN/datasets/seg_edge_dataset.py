import os
import random

import cv2
import pytorch_lightning as pl
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from parameters import setupArgs

import albumentations as A

transform = A.Compose([
    #
    # A.HorizontalFlip(p=0.6),  # 水平翻转，概率为 50%
    # A.VerticalFlip(p=0.6),
    A.Rotate(limit=150, p=0.5),  # 随机旋转图像，角度限制在 ±30 度，概率为 50%

    A.RandomResizedCrop(height=512, width=512, scale=(0.2, 0.8), p=0.5),

])


def apply_polarization_mode(image, mode):
    """
    Freeman 分量顺序：[Surface, Double-bounce, Volume]
    根据模式选择性保留通道
    """
    assert image.shape[2] == 3, "输入图像应为 Freeman 三通道"
    img = image.copy().astype(np.float32)

    if mode == "surface":  # 仅表面散射
        img[:, :, :2] = 0.0  # 保留 B 通道 (S)
    elif mode == "double":  # 仅二面角散射
        img[:, :, 1:] = 0.0  # 保留 R 通道 (D)
    elif mode == "volume":  # 仅体散射
        img[:, :, [0, 2]] = 0.0  # 保留 G 通道 (V)
    elif mode == "surface_double":  # 表面 + 二面角
        img[:, :, 1] = 0.0  # 保留 R,B 通道
    elif mode == "surface_volume":  # 表面 + 体散射
        img[:, :, 0] = 0.0  # 保留 G,B 通道
    elif mode == "double_volume":  # 二面角 + 体散射
        img[:, :, 2] = 0.0  # 保留 R,G 通道
    elif mode == "full":  # 全 Freeman 通道
        pass
    else:
        raise ValueError(f"未知模式：{mode}")
    return img

def extract_contours(label, line_thickness=2):
    """
    提取标签图像的边缘并返回边缘图像。

    :param label: 标签图像的NumPy数组
    :param line_thickness: 边缘线的粗细
    :return: 二值边缘图像的NumPy数组
    """
    # 查找轮廓
    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个黑色图像只保留白色轮廓
    edge_label = np.zeros_like(label)

    # 绘制轮廓
    cv2.drawContours(edge_label, contours, -1, (255), thickness=line_thickness)

    return edge_label

global_epoch = 0

class GF3_Train_dataset(Dataset):
    def __init__(self, train_image_list, train_label_list,mode="full"):
        self.num_classes = 2
        self.transform = transform
        self.image_list = train_image_list
        self.label_list = train_label_list
        self.mode = mode


    def __getitem__(self, item):
        image_path = self.image_list[item]
        label_path = self.label_list[item]

        assert image_path.split('\\')[-1].split('.')[0] == label_path.split('\\')[-1].split('.')[0]

        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path).convert('L'))

        image = apply_polarization_mode(image, self.mode)

        if self.transform:
            agument = self.transform(image=np.array(image), mask=np.array(label))
            image = agument['image'].transpose(2, 0, 1)
            label = agument['mask']
        normalized_image = (image / 255.0).astype(np.float32)

        one_hot_label = self.make_one_hot(label)
        edge_label = extract_contours(label)

        return normalized_image, label, one_hot_label, image,edge_label

    def make_one_hot(self, label):
        label = np.array(label)
        label[label >= (self.num_classes - 1)] = self.num_classes - 1
        # 生成one-hot标签
        flattened_label = label.reshape([-1])
        identity_matrix = np.eye(self.num_classes)
        one_hot_label = identity_matrix[flattened_label]
        one_hot_label = one_hot_label.reshape((label.shape[0], label.shape[1], self.num_classes))
        one_hot_label = one_hot_label.transpose(2, 0, 1).astype(np.int32)
        return one_hot_label

    def __len__(self):
        return len(self.image_list)


class GF3_Validation_dataset(Dataset):
    def __init__(self, validation_image_list, validation_label_list,mode="full"):
        self.num_classes = 2
        self.image_list = validation_image_list
        self.label_list = validation_label_list

        self.mode = mode

    def __getitem__(self, item):
        image_path = self.image_list[item]
        label_path = self.label_list[item]
        assert image_path.split('\\')[-1].split('.')[0] == label_path.split('\\')[-1].split('.')[0]

        image = np.array(Image.open(image_path))

        label = np.array(Image.open(label_path).convert('L'))

        image = apply_polarization_mode(image, self.mode)

        normalized_image = ((image / 255.0).astype(np.float32)).transpose(2, 0, 1)

        one_hot_label = self.make_one_hot(label)
        edge_label = extract_contours(label)

        return normalized_image, label, one_hot_label, image, edge_label


    def make_one_hot(self, label):
        label = np.array(label)
        label[label >= (self.num_classes - 1)] = self.num_classes - 1
        # 生成one-hot标签
        flattened_label = label.reshape([-1])
        identity_matrix = np.eye(self.num_classes)
        one_hot_label = identity_matrix[flattened_label]
        one_hot_label = one_hot_label.reshape((label.shape[0], label.shape[1], self.num_classes))
        one_hot_label = one_hot_label.transpose(2, 0, 1).astype(np.int32)
        return one_hot_label

    def __len__(self):
        return len(self.image_list)


class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if self.opt.augmentation:
            # self.train_data_path = r"C:\polsar\datasets\radasat2号polsar数据\aug1\train"
            self.train_data_path = r"C:\polsar\datasets\高分3号Polsar数据\aug\train"

            # self.train_data_path = r"C:\polsar\datasets\高分3polsar数据_wishart\aug\train"
        else:
            # self.train_data_path = r"C:\polsar\datasets\radasat2号polsar数据\lee_dataset\train"
            self.train_data_path = r"C:\polsar\datasets\高分3号Polsar数据\lee_dataset\train"
            # self.train_data_path = r"C:\polsar\datasets\高分3polsar数据_wishart\lee_dataset\train"

        # self.validation_dataset_path = r"C:\polsar\datasets\radasat2号polsar数据\lee_dataset\test"
        # self.validation_dataset_path = r"C:\polsar\datasets\高分3号Polsar数据\lee_dataset\test"
        # self.validation_dataset_path = r"C:\polsar\datasets\高分3polsar数据_wishart\lee_dataset\test"
        self.validation_dataset_path = r"C:\polsar\datasets\高分3号Polsar数据\aug\test"

        self.mode = "full"


        train_image_list, train_label_list, validation_image_list, validation_label_list = self.read_image()
        self.train_dataset = GF3_Train_dataset(train_image_list, train_label_list,mode=self.mode)
        self.validation_dataset = GF3_Validation_dataset(validation_image_list, validation_label_list,mode=self.mode)

    def read_image(self):
        train_image_list = [os.path.join(self.train_data_path + r'\image', image) for image in
                            os.listdir(self.train_data_path + r'\image')]
        train_label_list = [os.path.join(self.train_data_path + r'\label', label) for label in
                            os.listdir(self.train_data_path + r'\label')]
        validation_image_list = [os.path.join(self.validation_dataset_path + r'\image', image) for image in
                                 os.listdir(self.validation_dataset_path + r'\image')]
        validation_label_list = [os.path.join(self.validation_dataset_path + r'\label', label) for label in
                                 os.listdir(self.validation_dataset_path + r'\label')]
        return train_image_list, train_label_list, validation_image_list, validation_label_list

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.opt.bs,
                          shuffle=True, num_workers=0, drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.opt.bs,
                          shuffle=False, num_workers=0,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=1,
                          shuffle=False, num_workers=0,
                          pin_memory=True)


if __name__ == '__main__':
    opt = setupArgs()
    dataModule = DataModule(opt)
    train_dataloader = dataModule.train_dataloader()
    for i, data in enumerate(train_dataloader):
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(data[3].shape)
        print(data[3])

        break
    # val_dataloader = dataModule.val_dataloader()
    # for i, data in enumerate(val_dataloader):
    #     print(data[0].shape)
    #     print(data[1].shape)
    #     print(data[2].shape)
    #     print(data[3].shape)
    #     print(data[4].shape)
    #
    #     break
    # test_dataloader = dataModule.test_dataloader()
    # for i, data in enumerate(test_dataloader):
    #     print(data[0].shape)
    #     print(data[1].shape)
    #     print(data[2].shape)
    #     print(data[3].shape)
    #     print(data[4].shape)
    #     break
    #