import torch
import numpy as np

import os
from torch.nn import functional as F
from PIL import Image
import model
import cv2

def get_filelist(dir, Filelist):

    if os.path.isfile(dir):

        Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)

    return Filelist

#####################################

input_size = [256, 256, 3]

NUM_CLASSES = 2


use_gpu=True
if use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device=torch.device("cpu")


net = model.FCSSN(in_channels=input_size[2],out_channels=NUM_CLASSES)

net.to(device)

net.load_state_dict(torch.load(r"E:\IDUDL\pth\IDUDL.pth"))

def main(inputpath, namep):

    image = Image.open(inputpath)
    image = image.resize((input_size[0],input_size[1]))
    image = np.array(image)

    if image.ndim == 2:
        image = np.array([image])
    elif image.ndim >= 3:
        image = np.transpose(image, [2, 0, 1])
    else:
        raise SystemExit('please input correct size, e.g.(width,height)、(width,height,channel)')


    data = torch.from_numpy(np.array([image.astype('float32') / 255.])).to(device)

    out_result = net(data)
    out_result = F.softmax(out_result, dim=1)
    if use_gpu:
        out_result = out_result.cpu().detach().numpy()
    else:
        out_result = out_result.detach().numpy()
    out_result = out_result.reshape((NUM_CLASSES, input_size[0], input_size[1]))


    out_result = np.argmax(out_result, 0) * 255
    cv2.imwrite(r"E:\IDUDL\data\%s.png" % namep, out_result)



def mainf():
    dir=r'E:\IDUDL\data\demo\\'
    LIst=get_filelist(dir, [])
    print(LIst)
    print(len(LIst))
    ST=0

    for ii in range(len(LIst)):
        name=LIst[ii][20:]
        print(name)
        namep=name[:-4]
        print(namep)
        main(dir+name, namep)



if __name__ == '__main__':
    #run()
    mainf()


