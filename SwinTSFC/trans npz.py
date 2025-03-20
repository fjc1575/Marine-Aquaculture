import glob
from random import random
# 先转成黑白 再存npz
import cv2
import numpy as np
import os
from skimage import io
'''
def npzs(im, la, s):
    images_path = im
    labels_path = la
    path2 = s
    images = os.listdir(labels_path)
    for s in images:
        image_path1 = os.path.join(images_path, '21_0210', s)
        image_path2 = os.path.join(images_path, '22_1110', s)
        image_path3 = os.path.join(images_path, '23_0107', s)
        image_path4 = os.path.join(images_path, '23_0503', s)
        image_path5 = os.path.join(images_path, '23_0404', s)
        image_path6 = os.path.join(images_path, '23_0729', s)
        # image_path1 = os.path.join(image_path1, s)
        # image_path2 = os.path.join(image_path2, s)
        # image_path3 = os.path.join(image_path3, s)
        # print(image_path1)
        # print(image_path2)
        # print(image_path3)
        label_path = os.path.join(labels_path, s)

        image1 = io.imread(image_path1)
        image2 = io.imread(image_path2)
        image3 = io.imread(image_path3)
        image4 = io.imread(image_path4)
        image5 = io.imread(image_path5)
        image6 = io.imread(image_path6)
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		# 标签由三通道转换为单通道
        label = io.imread(label_path)
        # 保存npz文件
        np.savez(path2+s[:-4]+".npz",image1=image1,image2=image2,image3=image3,image4=image4,image5=image5,image6=image6,label=label)
        # np.savez(path2+s[:-4]+".npz", image=image, label=label)
'''
# npz('E:/remote/longtime/data/train/image/', 'E:/remote/longtime/data/train/label', './data/Synapse/traint_npz/')
# npz('E:/remote/longtime/data/test/image/', 'E:/remote/longtime/data/test/label', './data/Synapse/test7_vol_h5/')
# random   ALT+enter  自动导包
# npz('./img_datas/train/image3/', './img_datas/train/label3/', './data/Synapse/traint_npz/')
# npzs(r'E:\remote\guangdong\datasat/', r'E:\remote\guangdong\datasat\target\black/', './data/Synapse/trainl_512_0404/')
#
# def add(x: int, y: int):
#     return x+y
# add()   ctrl+p 显示方法的传参类型

def npz(im, la, s):
    images_path = im
    labels_path = la
    path2 = s
    images = os.listdir(images_path)
    for s in images:
        image_path = os.path.join(images_path, s)
        label_path = os.path.join(labels_path, s)
        print(s)
        image = io.imread(image_path)
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		# 标签由三通道转换为单通道
        label = io.imread(label_path)
        # 保存npz文件
        # np.savez(path2+s[:-4]+".npz", image1=image, image2=image, image3=image, label=label)
        np.savez(path2+s[:-4]+".npz", image=image, label=label)

# npz('E:/remote/longtime/data/train/image/', 'E:/remote/longtime/data/train/label', './data/Synapse/traint_npz/')
# npz('E:/remote/longtime/data/test/image/', 'E:/remote/longtime/data/test/label', './data/Synapse/test7_vol_h5/')
# random   ALT+enter  自动导包
npz(r'E:\remote\changhaixian\dataset\20231003/', r'E:\remote\changhaixian\dataset\target\black/', './data/Synapse/changhaixian/trainu/')

'''
def npzs(im, la, s):
    images_path = im
    labels_path = la
    path2 = s
    images = os.listdir(labels_path)
    for s in images:
        image_path1 = os.path.join(images_path, '20180722', s)
        image_path2 = os.path.join(images_path, '20191202', s)
        image_path3 = os.path.join(images_path, '20200310', s)
        image_path4 = os.path.join(images_path, '20210617', s)
        image_path5 = os.path.join(images_path, '20220919', s)
        image_path6 = os.path.join(images_path, '20231104', s)
        image_path7 = os.path.join(images_path, '20231003', s)

        label_path = os.path.join(labels_path, s)

        image1 = io.imread(image_path1)
        image2 = io.imread(image_path2)
        image3 = io.imread(image_path3)
        image4 = io.imread(image_path4)
        image5 = io.imread(image_path5)
        image6 = io.imread(image_path6)
        image7 = io.imread(image_path7)
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		# 标签由三通道转换为单通道
        label = io.imread(label_path)
        # 保存npz文件
        np.savez(path2+s[:-4]+".npz",image1=image1,image2=image2,image3=image3,image4=image4,image5=image5,image6=image6,image7=image7,label=label)
        # np.savez(path2+s[:-4]+".npz", image=image, label=label)

npzs(r'E:\remote\changhaixian\dataset', r'E:\remote\changhaixian\dataset\target\black/', './data/Synapse/changhaixian/trainl/')
'''