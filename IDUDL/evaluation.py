import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

def whiteblack(img):
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if img[m][n] == 255:
                img[m][n] = 0
            else:
                img[m][n] = 1
    return img

def eval1(x,y):
    gt_mat = x.flatten()
    pre_mat = y.flatten()

    con = confusion_matrix(gt_mat, pre_mat)

    if con.shape == (1,1):
        acc = 1
    else:
        a = con[0,0]+con[1,1]
        b = con[1,0]+con[0,0]+con[1,1]+con[0,1]
        acc = a/b

    return acc

def eva(input_path_pre,input_path_gt):
    files = os.listdir(input_path_gt)
    S = 0
    for i in range(len(files)):
        name = str(files[i])
        namep = name[:-4]
        gt_mat = cv2.imread(input_path_gt + namep + '.png', 0)
        gt_mat = whiteblack(gt_mat)
        pre_png = cv2.imread(input_path_pre + namep + '.png', 0)
        pre_png = whiteblack(pre_png)
        acc = eval1(gt_mat, pre_png)

        S = S + acc

        mean_acc = S / (i + 1)

    return mean_acc
