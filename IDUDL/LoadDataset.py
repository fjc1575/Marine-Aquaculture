from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import torch
import numpy as np
#图片路径，标签路径，是否按照灰度图进行处理
def ProcessingResult(image_path, label_path,input_size,NUM_CLASSES,state):
    # 读取图片->调整大小为(256,256)->转成numpy
    image=Image.open(image_path)
    image=image.resize((input_size[0],input_size[1]))
    image=np.array(image)
    # 判断是否是灰度图，如果是灰度图 (width,height)->(channel,width,height) 否则 (width,height,channel)->(channel,width,height)
    if image.ndim==2:
        image=np.array([image])
    elif image.ndim>=3:
        image=np.transpose(image,[2,0,1])
    else:
        raise SystemExit('请输入正确的图片格式，例:(width,height)、(width,height,channel)')
    image = torch.FloatTensor(image) / 255

    # 读取标签图片->调整大小为(input_size[0],input_size[1])->转成numpy
    label=Image.open(label_path).convert('L')
    label=label.resize(([input_size[0],input_size[1]]))
    label=np.array(label)
    #把label中像素点的数值对应为类别
    if state=='train':
        label[label >=(NUM_CLASSES - 1)] = NUM_CLASSES - 1
        # 生成one-hot标签
        one_hot_label = np.eye(NUM_CLASSES)[label.reshape([-1])].reshape((input_size[0], input_size[1], NUM_CLASSES))
        #返回归一化之后的原始图像,标签，one-hot标签
        return image,label,one_hot_label
    else:
        label[label >= (NUM_CLASSES - 1)] = 255

        return image,label



class LoadDataset(Dataset):
    def __init__(self, img_path, label_path,input_size,NUM_CLASSES,state):
        path = []
        imgs = os.listdir(img_path)
        for img in imgs:
            path.append((str(img_path + "/" + img), (label_path + "/" + str(img.split(".")[0]) + ".png")))
        self.path = path
        self.input_size=input_size
        self.NUM_CLASSES=NUM_CLASSES
        self.state=state
    def __getitem__(self, item):
        image_path, label_path = self.path[item]
        return ProcessingResult(image_path, label_path,self.input_size,self.NUM_CLASSES,self.state)
    def __len__(self):
        return len(self.path)