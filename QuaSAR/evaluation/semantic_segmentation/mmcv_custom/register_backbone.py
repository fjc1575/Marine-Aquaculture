# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES
from models import VisionTransformer#,SimCLR

# import torchvision
# from simclr.modules.identity import Identity

@BACKBONES.register_module()
class VisionTransformer(VisionTransformer):
    def __init__(self,
                 patch_size,
                 embed_dim,
                 with_fpn=True,
                 frozen_stages=-1,
                 out_indices=[3, 5, 7, 11],
                 out_with_norm=False,
                 use_checkpoint=False,
                 **kwargs):
        super(VisionTransformer, self).__init__(
            patch_size=patch_size,
            embed_dim=embed_dim,
            **kwargs)
        self.patch_size = patch_size
        self.with_fpn = with_fpn
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint

        if not out_with_norm:
            self.norm = nn.Identity()

        if with_fpn and patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif with_fpn and patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
        else:
            logger = get_root_logger()
            logger.info('Build model without FPN.')

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(VisionTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            self.pos_drop.eval()

        for i in range(1, self.frozen_stages + 1):

            if i  == len(self.blocks):
                norm_layer = getattr(self, 'norm') #f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            self.apply(self._init_weights)
            logger = get_root_logger()
            if  os.path.isfile(pretrained):
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                logger.info(f"checkpoint path {pretrained} is invalid, we skip it and initialize net randomly")
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x = self.prepare_tokens(x)
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if i in self.out_indices:
                xp = self.norm(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())

        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])
        return tuple(features)


'''
@BACKBONES.register_module()
class SimCLR(SimCLR):

    def get_resnet(self, name="resnet18", pretrained=False):
        resnets = {
            "resnet18": torchvision.models.resnet18(pretrained=pretrained),
            "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]

    def __init__(self,
                 name='resnet18',
                 # frozen_stages=0,
                 projection_dim=64,
                 n_features=384):

        super().__init__()
        # super(SimCLR, self).__init__()

        self.dim = nn.Linear(512, 128*128, bias=True)
        # self.frozen_stages = frozen_stages
        self.encoder = self.get_resnet(name)

        self.encoder.requires_grad = False

        self.n_features = n_features
        self.norm = nn.Identity()
        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        # self.projector = nn.Sequential(
        #     nn.Linear(512, self.n_features, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(self.n_features, projection_dim, bias=False),
        # )
        #
        # self.projector.requires_grad = False

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(1, n_features, kernel_size=2, stride=2),
            nn.SyncBatchNorm(n_features),
            nn.GELU(),
            nn.ConvTranspose2d(n_features, n_features, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(1, n_features, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Identity()
        # nn.Sequential(
        #     nn.Identity(),
        #     nn.ConvTranspose2d(1, n_features, kernel_size=2, stride=2),
        # )

        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #     nn.Sequential(
        #     nn.ConvTranspose2d(1, n_features, kernel_size=2, stride=2),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SimCLR, self).train(mode)
        # self._freeze_stages()

    # def _freeze_stages(self):
    #     if self.frozen_stages >= 0:
    #         self.encoder.eval()
    #         for param in self.encoder.parameters():
    #             param.requires_grad = False
    #         # self.encoder.fc.requires_grad = False
    #         self.projector.requires_grad = False
            # self.pos_drop.eval()

        # for i in range(1, self.frozen_stages + 1):
        #
        #     if i == len(self.Block):
        #         norm_layer = getattr(self, 'norm') #f'norm{i-1}')
        #         norm_layer.eval()
        #         for param in norm_layer.parameters():
        #             param.requires_grad = False
        #
        #     m = self.Block[i - 1]
        #     m.eval()
        #     for param in m.parameters():
        #         param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            self.apply(self._init_weights)
            logger = get_root_logger()
            if os.path.isfile(pretrained):
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                logger.info(f"checkpoint path {pretrained} is invalid, we skip it and initialize net randomly")
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x_i):#, x_j

        B, _, H, W = x_i.shape # 2, 3, 512, 512
        feature = []

        h_i = self.encoder(x_i) # 2, 512
        # print(h_i.shape)
        # z_i = self.projector(h_i)
        # print(z_i.shape)
        # print(h_i.shape)
        h_j = self.dim(h_i) # 2, 262144
        h_j = h_j.unsqueeze(0) #1, 2, 262144

        h_j = self.norm(h_j).permute(0, 2, 1).reshape(B, -1, 128, 128) #h_j[:, 1:, :]

        # 2,512,1
        # print(h_j.shape)# 2, 1, 512, 512
        feature.append(h_j)#.permute(2, 3, 0, 1)
        # print(feature[0].shape)
        feature.append(h_j)#.permute(2, 3, 0, 1)
        # print(feature[1].shape)
        feature.append(h_j)#.permute(2, 3, 0, 1)
        # print(feature[2].shape)
        feature.append(h_j) #512, 512, 2, 1.permute(2, 3, 0, 1)
        # print(feature[3].shape)
        # import time
        # time.sleep(1000)
        # z_i = self.projector(h_i) # 2, 64
        # z_j = self.projector(h_j)
        # print(z_i.shape)
        # z_i = self.norm(z_i[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, H, W)
        # feature.append(z_i)

        # print(feature.shape)

        # ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        # for i in range(len(feature)):
        #     feature[i] = ops[i](feature[i])
        feature[0] = self.fpn1(feature[0])
        feature[1] = self.fpn2(feature[1])
        feature[2] = self.fpn3(feature[0])
        feature[3] = self.fpn4(feature[0])

        # 2,384,128,128 ViT
        # print(feature[0].shape)#torch.Size([512, 512, 8, 4])
        # for zz in range(len(feature)):
        #     print('%s='%zz, feature[zz].shape)

        return tuple(feature)#, z_i #, z_j, h_j

    # features = []
    # for i, blk in enumerate(self.blocks):
    #
    #     x = blk(x)
    #     if i in self.out_indices:
    #         xp = self.norm(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp)
    #         features.append(xp.contiguous())
    #
    # if self.with_fpn:
    #     ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
    #     for i in range(len(features)):
    #         features[i] = ops[i](features[i])
'''
