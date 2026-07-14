import random
import time
import warnings
from pprint import pprint
import copy
import pytorch_lightning as pl
from lightning_fabric import seed_everything
# from pytorch_grad_cam import GradCAM, LayerCAM, XGradCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch
from torch import nn
import gc

from CAM.grad_cam_training import GradCAM
from architectures.lfi import BPCAMNET
from loss.loss import FocalLoss_Edge, BceIoULoss, CrossEntropyLoss, ScatteringPenaltyLoss

import numpy as np


from datasets.seg_edge_dataset import DataModule

from metric.score import Evaluator


from parameters import setupArgs
from utils.cuda import setupCuda
from utils.save_image import save_test_image

from utils.seed import setSeed
from utils.wandbExec import wandb_init



class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, pred, target):
        # 对每个热力图应用 softmax
        pred = F.log_softmax(pred.view(pred.size(0), -1), dim=-1)  # 先 log_softmax
        target = F.softmax(target.view(target.size(0), -1), dim=-1)  # 然后 softmax

        # 计算 KL 散度
        loss = F.kl_div(pred, target, reduction='batchmean')

        return loss


class Supervision_Train(pl.LightningModule):
    def __init__(self, opt,loss=None):
        super().__init__()

        self.opt = opt
        # self.opt.arch = 'bpcamnet'

        self.net = BPCAMNET()


        self.loss = CrossEntropyLoss()
        self.loss_edge =  BceIoULoss(weight=torch.tensor(0.1))
        self.loss_contrastive_criterion = loss
        # self.loss_contrastive_criterion = nn.L1Loss()
        # self.loss_contrastive_criterion = nn.MSELoss()
        # self.loss_contrastive_criterion = CosineSimilarityLoss()
        # self.loss_contrastive_criterion = KLDivergenceLoss()
        self.train_loss_collect = []
        self.train_loss_con = []
        self.val_loss_collect = []
        self.test_image_dict_list = []

        self.metrics_train = Evaluator(num_class=self.opt.num_class)
        self.metrics_val = Evaluator(num_class=self.opt.num_class)

        self.val_label_list = []


        self.start_time = None
        self.end_time = None

        print('当前的gamma值为：',self.opt.gamma)

    def forward(self, x):

        if self.training:

            seg_pre, edge,out,res4 = self.net(x)

            return seg_pre,edge,out,res4

        else:

            seg_pre = self.net(x)

            return seg_pre

    def on_train_epoch_start(self):

        self.start_time = time.time()

        optimizer = self.trainer.optimizers[0]  # 获取优化器对象
        self.learning_rate = optimizer.param_groups[0]['lr']  # 获取当前学习率
        print(f'Current learning rate: {self.learning_rate}')
        if self.opt.log_online:
            wandb.log({'lr': self.learning_rate},step=self.trainer.current_epoch)

    def training_step(self, batch, batch_idx):

        input, label, label_one_hot, rgb_image,edge_label = batch

        edge_label[edge_label>0] = 1

#############第一次前向传播######################

        self.net.training = True

        seg,edge,out,res4 = self.forward(input)


        mask = torch.argmax(label_one_hot, dim=1).cpu().numpy()

        pre_mask = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)

        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i], pre_mask[i].cpu().numpy())

        loss_out = self.loss(out, torch.tensor(mask).to('cuda'))

        loss_seg = self.loss(seg, torch.tensor(mask).to('cuda'))

        loss_edge = self.loss_edge(torch.sigmoid(edge), edge_label.unsqueeze(1))

        loss_stage1 = loss_seg + loss_edge + loss_out

        #############计算CAM######################
        model_copy = copy.deepcopy(self.net)
        if self.current_epoch>=0:
            target_layer = [model_copy.backbone.layer4[-1]]
            grad_cam = torch.from_numpy(self.generate_cam(model_copy,target_layer, input, pre_mask.cpu().detach().numpy())).to('cuda')

            feature_cam = torch.sum(res4,dim=1)
            min_val = feature_cam.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]  # 每个通道的最小值
            max_val = feature_cam.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]  # 每个通道的最大值
            feature_cam = (feature_cam - min_val) / (max_val - min_val + 1e-8)


            loss_contrastive = self.loss_contrastive_criterion(grad_cam, feature_cam)

            if torch.isnan(loss_contrastive):
                print("NaN loss detected, skipping this batch")
                return None

            self.train_loss_con.append(loss_contrastive.item())


            print(loss_contrastive.item())
        else:
            loss_contrastive = 0

        loss = loss_stage1 + self.opt.gamma*loss_contrastive

        self.train_loss_collect.append(loss.item())
#######################################################################
        return {"loss": loss}

    def on_train_epoch_end(self):


        loss = np.mean(self.train_loss_collect)
        loss_con = np.mean(self.train_loss_con)
        Precision = self.metrics_train.Precision()[1]
        IoU = self.metrics_train.Intersection_over_Union()[1]
        Miou = self.metrics_train.Mean_Intersection_over_Union()
        F1 = self.metrics_train.F1()[1]
        OA = self.metrics_train.OA()
        Recall = self.metrics_train.Recall()[1]
        Kappa = self.metrics_train.kappa()

        self.metrics_train.reset()
        self.train_loss_collect = []
        self.train_loss_con = []

        train_res = {'train_IoU': IoU,
                     'train_F1': F1,
                     'train_OA': OA,
                     'train_Recall': Recall,
                     'train_Kappa': Kappa,
                     'train_miou': Miou,
                     'train_Precision': Precision,
                     'train_loss': loss,
                     'train_loss_con':loss_con}

        print("Train Results:")
        pprint(train_res)
        self.log_dict(train_res)
        if self.opt.log_online:
            wandb.log(train_res,step=self.trainer.current_epoch)


        self.end_time = time.time()

        print(f"Epoch {self.trainer.current_epoch} took {self.end_time - self.start_time} seconds.")

    def generate_cam(self, model, target_layer, inputs, pre_mask):
        cam_list = []
        for input, sub_mask in zip(inputs, pre_mask):
            targets = [SemanticSegmentationTarget(0, sub_mask)]
            with GradCAM(model=model, target_layers=target_layer, use_cuda=torch.cuda.is_available(), pre_mask=sub_mask
                         ) as cam:
                grayscale_cam = cam(input_tensor=input.unsqueeze(0), targets=targets)[0]
                cam_list.append(grayscale_cam)
        return np.array(cam_list)


    def validation_step(self, batch, batch_idx):

        input, label, label_one_hot, rgb_image,edge_label = batch

        output,_ = self.forward(input)

        mask = torch.argmax(label_one_hot, dim=1).cpu().numpy()

        pre_mask = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)

        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i], pre_mask[i].cpu().numpy())

        loss = self.loss(output, torch.tensor(mask).to('cuda'))

        self.val_loss_collect.append(loss.item())


        return {"loss_val": loss}

    def on_validation_epoch_end(self):



        loss = np.mean(self.val_loss_collect)
        Precision = self.metrics_val.Precision()[1]
        IoU = self.metrics_val.Intersection_over_Union()[1]
        Miou = self.metrics_val.Mean_Intersection_over_Union()
        F1 = self.metrics_val.F1()[1]
        OA = self.metrics_val.OA()
        Recall = self.metrics_val.Recall()[1]
        Kappa = self.metrics_val.kappa()

        self.metrics_val.reset()
        val_res = {'val_IoU': IoU,
                   'val_F1': F1,
                   'val_OA': OA,
                   'val_Recall': Recall,
                   'val_Kappa': Kappa,
                   'val_miou': Miou,
                   'val_Precision': Precision,
                   'val_loss': loss}
        print("Val Results:")
        pprint(val_res)
        self.log_dict(val_res)
        if self.opt.log_online:
            wandb.log(val_res,step=self.trainer.current_epoch)

    def test_step(self, batch, batch_idx):

        input, label, label_one_hot, rgb_image,edge_label = batch

        label[label > 0] = 255

        output = self.forward(input)

        mask = torch.argmax(label_one_hot, dim=1).cpu().numpy()

        pre_mask = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1).cpu().numpy()

        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i], pre_mask[i])


        test_image_dict = {'image': np.array(rgb_image.cpu()),
                           'label': np.array(label.cpu()),
                           'mask': np.array(pre_mask*255).astype(np.uint8)}
        self.test_image_dict_list.append(test_image_dict)



    def on_test_epoch_end(self):

            save_test_image(self.test_image_dict_list, self.opt)


            Precision = self.metrics_val.Precision()[1]
            IoU = self.metrics_val.Intersection_over_Union()[1]
            MioU = self.metrics_val.Mean_Intersection_over_Union()
            F1 = self.metrics_val.F1()[1]
            OA = self.metrics_val.OA()
            Recall = self.metrics_val.Recall()[1]
            Kaapa = self.metrics_val.kappa()

            self.metrics_val.reset()
            self.test_image_dict_list = []
            test_res = {'test_IoU': IoU, 'test_F1': F1, 'test_OA': OA, 'test_Recall': Recall, 'test_Kappa': Kaapa,
                        'test_miou': MioU, 'test_Precision': Precision}
            print('test_res:', test_res)
            self.log_dict(test_res)
            if self.opt.log_online:
                wandb.log(test_res)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.opt.lr, weight_decay=1e-5)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.opt.n_epochs, eta_min=0)
        return [optimizer], [lr_scheduler]


# training
def main(opt,loss):

    setSeed(opt)
    setupCuda(opt)
    if opt.log_online:
        wandb_init(opt)

    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor='val_OA',
                                          save_last=True,
                                          mode='max',
                                          dirpath=opt.model_save_path + '\\' + opt.arch + '_epoch' + str(opt.n_epochs),
                                          filename=opt.arch)

    model = Supervision_Train(opt,loss)


    dataModule = DataModule(opt)

    seed_everything(opt.seed)
    trainer = pl.Trainer(devices=opt.gpu,
                         max_epochs=opt.n_epochs,
                         accelerator='auto',
                         strategy='auto',
                         callbacks=[checkpoint_callback],)
    trainer.fit(model=model,datamodule=dataModule)

    trainer.test(model=model,datamodule=dataModule)


if __name__ == "__main__":

    loss_list = [
        nn.MSELoss(),
        KLDivergenceLoss(),
        # nn.L1Loss()
    ]

    gamma_list = [0.01,2,5]

    warnings.filterwarnings("ignore")

    opt = setupArgs()
    for gamma in gamma_list:
        opt.gamma = gamma
        opt.arch = 'lfi' + str(gamma)
        main(opt,nn.MSELoss())
        wandb.finish()
        torch.cuda.empty_cache()
