import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # if random.random() > 0.5:   # 翻转 数据增强
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)
    
    
    def __getitem__(self, idx):    # 多输入
        if self.split == "trainl_ch":   # npz
            # img_path_list = self.img_list[idx]
            datas = []
            slice_name = self.sample_list[idx].strip('\n')
            # print(slice_name)
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            # print(data_path)
            data = np.load(data_path)
            image1, image2, image3, image4, image5, image6, label = data['image1'],data['image2'],data['image3'],data['image4'],data['image5'],data['image6'],data['label']
            # image1, image2, image3, image4, image5, image6, image7, label = data['image1'],data['image2'],data['image3'],data['image4'],data['image5'],data['image6'],data['image7'],data['label']
            image1 = torch.from_numpy(image1.astype(np.float32))
            image1 = torch.unsqueeze(image1, dim=0)
            image2 = torch.from_numpy(image2.astype(np.float32))
            image2 = torch.unsqueeze(image2, dim=0)
            image3 = torch.from_numpy(image3.astype(np.float32))
            image3 = torch.unsqueeze(image3, dim=0)
            image4 = torch.from_numpy(image4.astype(np.float32))
            image4 = torch.unsqueeze(image4, dim=0)
            image5 = torch.from_numpy(image5.astype(np.float32))
            image5 = torch.unsqueeze(image5, dim=0)
            image6 = torch.from_numpy(image6.astype(np.float32))
            image6 = torch.unsqueeze(image6, dim=0)
            # image7 = torch.from_numpy(image7.astype(np.float32))
            # image7 = torch.unsqueeze(image7, dim=0)

            datas.append(image1)
            datas.append(image2)  # ch 是 image3
            datas.append(image3)  # ch 是 image2
            datas.append(image4)
            datas.append(image5)
            datas.append(image6)
            # datas.append(image7)
            datas = torch.cat(datas, dim=0)
            # print(datas)

            sample = {'image': datas, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            sample['case_name'] = self.sample_list[idx].strip('\n')
            return sample
        else:
            datas = []
            slice_name = self.sample_list[idx].strip('\n')
            # print('slice_name', slice_name)
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            # print(data_path)
            data = np.load(data_path)
            image1, image2, image3, image4, image5, image6, label = data['image1'],data['image2'],data['image3'],data['image4'],data['image5'],data['image6'],data['label']
            # image1, image2, image3, image4, image5, image6, image7, label = data['image1'],data['image2'],data['image3'],data['image4'],data['image5'],data['image6'],data['image7'],data['label']
            image1 = torch.from_numpy(image1.astype(np.float32))
            image1 = torch.unsqueeze(image1, dim=0)
            image2 = torch.from_numpy(image2.astype(np.float32))
            image2 = torch.unsqueeze(image2, dim=0)
            image3 = torch.from_numpy(image3.astype(np.float32))
            image3 = torch.unsqueeze(image3, dim=0)
            image4 = torch.from_numpy(image4.astype(np.float32))
            image4 = torch.unsqueeze(image4, dim=0)
            image5 = torch.from_numpy(image5.astype(np.float32))
            image5 = torch.unsqueeze(image5, dim=0)
            image6 = torch.from_numpy(image6.astype(np.float32))
            image6 = torch.unsqueeze(image6, dim=0)
            # image7 = torch.from_numpy(image7.astype(np.float32))
            # image7 = torch.unsqueeze(image7, dim=0)

            datas.append(image1)
            datas.append(image2)
            datas.append(image3)
            datas.append(image4)
            datas.append(image5)
            datas.append(image6)
            # datas.append(image7)
            datas = torch.cat(datas, dim=0)
            # print(datas)

            sample = {'image': datas, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            sample['case_name'] = self.sample_list[idx].strip('\n')
            return sample
    
    '''
    def __getitem__(self, idx):  # 单输入
        if self.split == "train7":
            slice_name = self.sample_list[idx].strip('\n')
            # print(slice_name)
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # print("label: ", label)
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            #改，numpy转tensor
            image = torch.from_numpy(image.astype(np.float32))
            image = image.permute(2, 0, 1)
            label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    '''
    




