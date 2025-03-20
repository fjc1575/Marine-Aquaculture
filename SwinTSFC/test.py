import argparse
import logging
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
# from trainer import trainer_synapse5
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data/Synapse', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', default='./output_0408', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=151, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./pre_7', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default=r'/root/autodl-tmp/Swin-Unet-main/configs/swin_tiny_patch4_window7_224_lite.yaml',
                    metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "testl_512_0404")  # npz  testl_512_0404  testl_ch
config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="testl_512", list_dir=args.list_dir)   # txt   testl_ch
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    sum_ssim = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # print(sampled_batch["image"].size())   # [1, 3, 224, 224, 3] [b, t, h, w, c]
        h, w = sampled_batch["image"].size()[2:4]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i, mean_ssim = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        sum_ssim += mean_ssim
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_ssim %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], mean_ssim))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_ssim = sum_ssim / 40
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_ssim : %f' % (performance, mean_hd95, mean_ssim))
    return "Testing Finished!"

# 车道线 指标
def evaluate_model(model):
    model.eval()
    i = 0
    precision = 0.0
    recall = 0.0
    test_loss = 0
    correct = 0
    error = 0
    db_test = args.Dataset(base_dir=args.volume_path, split="testl_512", list_dir=args.list_dir)   # txt
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    with torch.no_grad():
        for sample_batched in test_loader:
            i += 1
            data, target = sample_batched["image"], sample_batched["label"].type(torch.LongTensor)
            # data, target = sample_batched["image"].to(device), sample_batched["label"].type(torch.LongTensor)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # print('device', device)
            # data, target = sample_batched["image"].to(device), sample_batched["label"].type(torch.LongTensor).to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # 返回两个，一个是最大值另一个是最大值索引
            img = torch.squeeze(pred).cpu().numpy()*255
            # img = torch.squeeze(pred).cuda().numpy()*255
            lab = torch.squeeze(target).cpu().numpy()*255
            # lab = torch.squeeze(target).cuda().numpy()*255
            img = img.astype(np.uint8)
            lab = lab.astype(np.uint8)
            kernel = np.uint8(np.ones((3, 3)))

            #accuracy
            # test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            #precision,recall,f1
            label_precision = cv2.dilate(lab, kernel)
            pred_recall = cv2.dilate(img, kernel)
            img = img.astype(np.int32)
            lab = lab.astype(np.int32)
            label_precision = label_precision.astype(np.int32)
            pred_recall = pred_recall.astype(np.int32)
            a = len(np.nonzero(img*label_precision)[1])
            b = len(np.nonzero(img)[1])
            if b == 0:
                error = error+1
                continue
            else:
                precision += float(a/b)
            c = len(np.nonzero(pred_recall*lab)[1])
            d = len(np.nonzero(lab)[1])
            if d == 0:
                error = error + 1
                continue
            else:
                recall += float(c / d)
            F1_measure = (2*precision*recall)/(precision+recall)

    test_loss /= (len(test_loader.dataset) / args.batch_size)
    test_acc = 100. * int(correct) / (len(test_loader.dataset) * 512 * 512)
    print('\n Accuracy: {}/{} ({:.5f}%)'.format(
         int(correct), len(test_loader.dataset), test_acc))

    precision = precision / (len(test_loader.dataset) - error)
    recall = recall / (len(test_loader.dataset) - error)
    F1_measure = F1_measure / (len(test_loader.dataset) - error)
    print('Precision: {:.5f}, Recall: {:.5f}, F1_measure: {:.5f}\n'.format(precision, recall, F1_measure))


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True


    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cpu()

    snapshot = os.path.join(args.output_dir, 'best_model.pth')   # ./output/best_model.pth
    # snapshot = os.path.join(args.output_dir, 'high', 'best_model.pth')   # ./output/high/best_model.pth
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    print(snapshot)
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet", msg)

    # snapshot_name = snapshot.split('/')[-1]
    snapshot_name = 'swinl_sc'   # 后加
    # log_folder = './test_log/test_log_'
    log_folder = './output_0408/swinl_sc'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "swinl_sc")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)

    # use_cuda = args.cuda and torch.cuda.is_available()  # 指标
    # device = torch.device("cpu" if use_cuda else "cpu")   # 指标
    evaluate_model(net)  # 指标


