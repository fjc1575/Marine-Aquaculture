import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from utils import BoundaryLoss
from utils import TemporalConsistencyLoss
from torchvision import transforms
from utils import test_single_volume

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/swinl_512", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train7",
    #                             transform=transforms.Compose(
    #                             [RandomGenerator(output_size=[args.img_size, args.img_size])]))  # swin 数据增强
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="trainl_512",
                             transform=None)      # txt  trainl_512  trainl_ch
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    tem_loss = TemporalConsistencyLoss()  # 后加
    boundary_Loss = BoundaryLoss()   # 后加
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log_swinl_512')   # log文件夹
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)   # 终端进度条
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
                # image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

                image_batch = torch.tensor(image_batch)  # longtime， 转换为tensor
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = model(image_batch)
                # print("outputs: ", outputs.shape)  # [4, 2, 224, 224]
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)  # 学习率
                writer.add_scalar('info/total_loss', loss, iter_num)
                writer.add_scalar('info/loss_ce', loss_ce, iter_num)  # loss可视化
                writer.add_scalar('info/loss_dice', loss_dice, iter_num)  # loss可视化
                # writer.add_scalar('info/loss_tem', loss_tem, iter_num)  # loss可视化
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
                # logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_tem: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_tem.item()))

                if iter_num % 20 == 0:
                    # image = image_batch[1, 0:1, :, :]  # swin  [1, 0:1, :, :]
                    image_batch = image_batch.permute(0, 4, 2, 3, 1)  # longtime
                    image = image_batch[0, 0:1, :, :, -1]  # longtime   [b, c, h, w, t]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    # labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                    labs = label_batch[0, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)


        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 50 and epoch_num % 10 ==0 :
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            # break

    writer.close()
    return "Training Finished!"