import os

import torch


#####设置cuda环境######
def setupCuda(opt):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

print(torch.cuda.is_available())