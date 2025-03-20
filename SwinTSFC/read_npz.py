import numpy as np
import torch
path = r"D:\pycharm\UNet\Swin-Unet-main\data\Synapse\changhaixian\trainl\0_0.npz"
data = np.load(path)
print(data)
# ac = np.load(path)
# print(ac)
# print(ac.files)
# print(ac['image'])
# print('------------------')
# print(ac['label'])
# tensor = torch.from_numpy(ac['label'])
# print("-------------")
# print(tensor)
# print(tensor.shape)