import torch

from architectures import PCAMNet


#选择模型
def selectArch(opt):
    if opt.arch == 'segnet':
        return None
    elif opt.arch == 'pcamnet':
        if opt.bin == 2:
            return PCAMNet.PCAMNet(num_classes=opt.num_class,bin_size = [(2, 2), (2, 2), (2, 2), (2,2)])
        elif opt.bin == 4:
            return PCAMNet.PCAMNet(num_classes=opt.num_class,bin_size =[(4, 4), (4, 4), (4, 4), (4, 4)])
        elif opt.bin == 8:
            return PCAMNet.PCAMNet(num_classes=opt.num_class, bin_size=[(8, 8), (8, 8), (8, 8), (8, 8)])
        else:
            return PCAMNet.PCAMNet(num_classes=opt.num_class, bin_size=[(1, 1), (1, 1), (1, 1), (1, 1)])


    else:
        raise Exception('Architecture undefined!')
