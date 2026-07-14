import torch
import random
import numpy as np
import torch.backends.cudnn

def setSeed(opt):
    torch.backends.cudnn.deterministic = True
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)