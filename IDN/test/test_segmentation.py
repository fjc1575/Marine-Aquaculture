import warnings

import pytorch_lightning as pl


from parameters import setupArgs


from utils.cuda import setupCuda

from utils.seed import setSeed
from utils.wandbExec import wandb_init




# training
def main():
    warnings.filterwarnings("ignore")
    # 初始化操作
    opt = setupArgs()
    opt.log_online = False
    setSeed(opt)
    setupCuda(opt)
    from train.train_bpcamnet import Supervision_Train
    from datasets.seg_edge_dataset import DataModule
    model = Supervision_Train.load_from_checkpoint(r'C:\polsar\code\PolSARSeg\test\bpcamnet-4444-gf3.ckpt',opt=opt)
    # model = Supervision_Train.load_from_checkpoint(r'C:\polsar\code\PolSARSeg - 副本\weights\lfi0.5.ckpt', opt=opt)
    dataModule = DataModule(opt)

    trainer = pl.Trainer()

    trainer.test(model=model,datamodule=dataModule)




if __name__ == "__main__":
   main()