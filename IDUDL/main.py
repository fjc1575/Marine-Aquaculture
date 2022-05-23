import cv2
import os
import argparse
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from skimage.util import img_as_float
from skimage.segmentation import slic
from sklearn.metrics import confusion_matrix

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import model
from evaluation import eva
from transform import trans
from LoadDataset import LoadDataset
from train_val_dataloader import train_loader1, newloader_FEN, label_to_img

torch.cuda.current_device()
use_cuda = torch.cuda.is_available()

use_gpu = True
if use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Incremental double unsupervised deep learning')#创建对象
parser.add_argument('--nChannel', default=100, type=int,
                    help='number of channels--FEN')
parser.add_argument('--number', default=3, type=int,
                    help='number of convolutional layers--FEN')
parser.add_argument('--outChannel', default=2, type=int,
                    help='number of class--FEN')
parser.add_argument('--inChannel', default=5, type=int,
                    help='number of input channels--FEN')
parser.add_argument('--inputDataPath', default=r'E:\IDUDL\data\image\\', type=str,
                    help='input dataset path--FEN')
parser.add_argument('--outputDataPath', default=r'E:\IDUDL\experiment\\', type=str,
                    help='output--IDUDL')
parser.add_argument('--FEN_epoch', default=5, type=int,
                    help='number of epoch--FEN')
parser.add_argument('--FCSSN_epoch', default=200, type=int,
                    help='number of epoch--FCSSN')
parser.add_argument('--FCSSN_max_epoch', default=500, type=int,
                    help='number of epoch--FCSSN')
parser.add_argument('--FCSSN_classes', default=2, type=int,
                    help='number of classes--FCSSN')
args = parser.parse_args()




def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
    dst = cv2.filter2D(img, -1, kernel=kernel)

    dst1 = cv2.bilateralFilter(src=dst, d=0, sigmaColor=100, sigmaSpace=15)
    return dst1

def BW(gray, inputimage):
    choice1 = np.where(inputimage == 255)
    black = np.mean(gray[choice1])
    choice2 = np.where(inputimage == 0)
    white = np.mean(gray[choice2])

    if black > white:
        for i in range(inputimage.shape[0]):
            for j in range(inputimage.shape[1]):
                if inputimage[i][j] == 255:
                    inputimage[i][j] = 0
                else:
                    inputimage[i][j] = 255
    return inputimage

def Dice_loss(inputs, target):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:

        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)
    result = temp_target*temp_inputs
    shang = torch.sum(result,dim=(1,2))
    xia_x = torch.sum(temp_inputs,dim=(1,2))
    xia_y = torch.sum(temp_target,dim=(1,2))
    score = 1-torch.mean(((2*shang)+1)/(xia_x+xia_y+1))
    return score

def writetxt(epoch,loss,accuracy):
    dataline=str(epoch)+" "+str(loss)+" "+str(accuracy)+"\n"
    with open("test.txt", "a") as f:
        f.write(dataline)

def get_filelist(dir, Filelist):

    newDir = dir

    if os.path.isfile(dir):

        Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)

    return Filelist

def main(input_path):

    idudl_round = 1

    while True:

################### FEN ######################
        files = os.listdir(input_path).copy()

        for z in range(len(files)):
            namep = str(files[z])[:-4]

            img = cv2.imread(input_path + str(files[z]))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data1 = trans(gray)

            transimg = sharpen(img)


            segments = slic(img_as_float(transimg), n_segments=30, sigma=5)  # gf-3 ---- 30  5   radarsat-2  120 3

            data = torch.from_numpy(np.array([data1.astype('float32') / 255.]))  # .transpose((2, 0, 1)), number=args.number

            if use_cuda:
                data = data.cuda()
            data = Variable(data)

            FEN = model.FEN(input_dim=data1.shape[0], channel=args.nChannel, out_channel=args.outChannel)
            if use_cuda:
                FEN.cuda()
            FEN.train()
            loss_sim = torch.nn.CrossEntropyLoss()

            loss_hpy = torch.nn.L1Loss(size_average=True)
            loss_hpz = torch.nn.L1Loss(size_average=True)

            HPy_target = torch.zeros(img.shape[0] - 1, img.shape[1], 100)
            HPz_target = torch.zeros(img.shape[0], img.shape[1] - 1, 100)
            if use_cuda:
                HPy_target = HPy_target.cuda()
                HPz_target = HPz_target.cuda()

            optimizer_sim = optim.SGD(FEN.parameters(), lr=0.1, momentum=0.9)

            label_colours = np.array([[255, 255, 255], [0, 0, 0]])

            for e_i in range(args.FEN_epoch):

                final = np.zeros(img. shape[0:2]) # for aquaculture

                optimizer_sim.zero_grad()

                output = FEN(data)
                out_sim = output

                output_sim = out_sim[0].permute(1, 2, 0).contiguous().view(-1, 2)

                # outputHP = output_sim.reshape((img.shape[0], img.shape[1], args.outChannel))

                # HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
                # HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
                # lhpy = loss_hpy(HPy, HPy_target)
                # lhpz = loss_hpz(HPz, HPz_target)

                ignore, target = torch.max(output_sim, 1)

                if idudl_round == 1:
                    im_target = target.data.cpu().numpy()
                else:
                    FCSSN_output = args.outputDataPath + '/%sFCSSNout/' % (idudl_round-1)
                    im_target = cv2.imread(FCSSN_output + str(files[z]),0)

                nLabels = len(np.unique(im_target))
                arr = im_target.flatten()

                arr_gb = pd.Series(arr)  # 转换数据类型
                arr_gb = arr_gb.value_counts()  # 计数
                arr_gb.sort_index(inplace=True)

                if True:
                    im_target_rgb = np.array([label_colours[c % 2] for c in im_target])

                    im_target_rgb = im_target_rgb.reshape((256, 256, 3)).astype(np.uint8)
                    cv2.imshow("output", im_target_rgb)
                    cv2.waitKey(10)

                hhh = im_target.reshape(256, 256)

                for (s_i, segVal) in enumerate(np.unique(segments)):
                    mask = np.zeros(img.shape[:2], dtype="uint8")
                    mask[segments == segVal] = 255

                    img1 = np.multiply(gray, mask)
                    label = np.multiply(hhh, mask)

                    choice = np.where(img1 != 0)
                    num1 = choice[0].shape[0]
                    new = img1 * label
                    zero = np.where(new != 0)
                    num2 = zero[0].shape[0]

                    if num1 != 0:
                        if num2 / num1 > 0.5:  # - step
                            for p_i in range(choice[0].shape[0]):
                                final[choice[0][p_i], choice[1][p_i]] = 255

                final = BW(gray, final)

                newloader_FEN(args.outputDataPath, namep, final, idudl_round)

                final = final.flatten()
                final = torch.from_numpy(final)
                final = final / 255

                if use_cuda:
                    final = final.cuda()
                final_label = final.long()

                if idudl_round == 1:
                    loss = 1 * loss_sim(output_sim, final_label) #+ 0.5 * (lhpy + lhpz)
                else:
                    loss = 1 * loss_sim(output_sim, final_label)

                loss.backward()
                optimizer_sim.step()

                print(e_i, '/', args.FEN_epoch, '|', ' label num :', nLabels, ' | loss :', loss.item())

################################ FCSSN #################################

        NUM_CLASSES = args.FCSSN_classes
        input_size = img.shape
        Batch_Size = 1

        trainDir, validDir = train_loader1(args.outputDataPath, idudl_round)
        label_to_img(args.inputDataPath, trainDir, validDir)

        train_image_path = trainDir[:-11] + 'trainimg'
        train_label_path = trainDir

        val_image_path = validDir[:-9] + 'valimg'
        val_label_path = validDir

        train_data = LoadDataset(train_image_path, train_label_path, input_size, NUM_CLASSES, 'train')
        val_data = LoadDataset(val_image_path, val_label_path, input_size, NUM_CLASSES, 'val')

        train_loader = DataLoader(dataset=train_data, batch_size=Batch_Size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=Batch_Size, shuffle=False)

        net = model.FCSSN(in_channels=input_size[2], out_channels=NUM_CLASSES)
        model.weights_init(net)

        net.to(device)

        optimizer1 = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=1, gamma=0.99)

        loss_func = torch.nn.CrossEntropyLoss()


        if idudl_round > 1:
            previous = idudl_round - 1
            input_path_previous = args.outputDataPath + '/%s/' % previous
            input_path_current = args.outputDataPath + '/%s/' % idudl_round
            acc = eva(input_path_previous, input_path_current)

            print('the pseudo-labels error:', (1 - acc))
        else:
            acc = 0

        if (idudl_round == 1) or (acc < 0.99):
            for tau in range(args.FCSSN_epoch):
                if True:
                    net.train()
                    all_loss = 0
                    with tqdm(total=math.ceil(len(train_data) / 1),
                              desc=f'Epoch{tau}/{args.FCSSN_epoch}', postfix=dict, mininterval=0.3) as pbar:
                        for sequence, (the_train_data, the_train_label, the_train_label_onehot) in enumerate(train_loader):
                            optimizer1.zero_grad()
                            out_result = net(the_train_data.to(device))

                            loss = loss_func(out_result, the_train_label.type(torch.long).to(device)) + Dice_loss(
                                out_result,
                                the_train_label_onehot.type(
                                    torch.long).to(
                                    device))
                            loss.backward()
                            optimizer1.step()

                            all_loss = all_loss + loss.item()
                            pbar.set_postfix(
                                {'stage': 'train', 'lr': optimizer1.state_dict()['param_groups'][0]['lr'],
                                 'loss': loss.item()})
                            pbar.update()
                        pbar.set_postfix(
                            {'stage': 'train', 'lr': optimizer1.state_dict()['param_groups'][0]['lr'],
                             'loss': all_loss / len(train_data)})
                        pbar.update()

                if True:
                    net.eval()
                    accuracy = 0
                    with torch.no_grad():
                        with tqdm(total=math.ceil(len(val_data) / 1),
                                  desc=f'Epoch{tau}/{args.FCSSN_epoch}', postfix=dict, mininterval=0.3) as pbar:
                            for sequence, (the_val_data, the_val_label) in enumerate(val_loader):
                                out_result = net(the_val_data.to(device))

                                out_result = F.softmax(out_result, dim=1)

                                out_result = out_result.cpu().detach().numpy()
                                out_result = out_result.reshape((NUM_CLASSES, input_size[0], input_size[1]))

                                out_result = np.argmax(out_result, 0) * 255
                                out_result = out_result.ravel()

                                the_val_label = the_val_label.detach().numpy()
                                the_val_label = the_val_label.reshape((256, 256))
                                the_val_label = the_val_label.ravel()
                                cm = confusion_matrix(the_val_label, out_result)

                                if cm.shape == (1, 1):
                                    this_sequence_accuracy = 0.5
                                else:
                                    this_sequence_accuracy = (cm[0][0] + cm[1][1]) / the_val_label.shape[0]
                                accuracy = accuracy + this_sequence_accuracy

                                pbar.set_postfix(
                                    {'stage': 'val', 'accuracy': this_sequence_accuracy})
                                pbar.update()

                            pbar.set_postfix(
                                {'stage': 'val', 'accuracy': accuracy / len(val_data)})
                            pbar.update()

                if tau == (args.FCSSN_epoch-1):
                    files1 = os.listdir(args.inputDataPath).copy()
                    for ii in range(len(files1)):
                        namep1 = str(files1[ii])[:-4]
                        image = Image.open(input_path + str(files1[ii]))
                        image = image.resize((input_size[0], input_size[1]))
                        image = np.array(image)

                        if image.ndim == 2:
                            image = np.array([image])
                        elif image.ndim >= 3:
                            image = np.transpose(image, [2, 0, 1])
                        else:
                            raise SystemExit('please enter correct size, e.g.(width,height)、(width,height,channel)')

                        data2 = torch.from_numpy(np.array([image.astype('float32') / 255.])).to(device)

                        final_out_result = net(data2)
                        final_out_result = F.softmax(final_out_result, dim=1)

                        if use_gpu:
                            final_out_result = final_out_result.cpu().detach().numpy()
                        else:
                            final_out_result = final_out_result.detach().numpy()
                        final_out_result = final_out_result.reshape((NUM_CLASSES, input_size[0], input_size[1]))

                        final_out_result = np.argmax(final_out_result, 0) * 255
                        cv2.imwrite(args.outputDataPath + '/%sFCSSNout' % idudl_round + '\%s.png' % namep1, final_out_result)

                    save_name = "Epoch" + str(tau) + "-train_loss-%.6f" % (
                                all_loss / len(train_data)) + "-val_accuracy-%.6f" % (
                                        accuracy / len(val_data)) + ".pth"
                    save_path = args.outputDataPath + '/%sFCSSNpth/' % idudl_round + save_name
                    torch.save(net.state_dict(), save_path)
                    scheduler.step()
                    writetxt(tau, all_loss / len(train_data), accuracy / len(val_data))

            idudl_round = idudl_round + 1
            continue

        if (idudl_round > 1) and (acc >= 0.99):
            for tau in range(args.FCSSN_max_epoch):
                if True:
                    net.train()
                    all_loss = 0
                    with tqdm(total=math.ceil(len(train_data) / 1),
                              desc=f'Epoch{tau}/{args.FCSSN_epoch}', postfix=dict, mininterval=0.3) as pbar:
                        for sequence, (the_train_data, the_train_label, the_train_label_onehot) in enumerate(
                                train_loader):
                            optimizer1.zero_grad()
                            out_result = net(the_train_data.to(device))

                            loss = loss_func(out_result, the_train_label.type(torch.long).to(device)) + Dice_loss(
                                out_result,
                                the_train_label_onehot.type(
                                    torch.long).to(
                                    device))
                            loss.backward()
                            optimizer1.step()

                            all_loss = all_loss + loss.item()
                            pbar.set_postfix(
                                {'stage': 'train', 'lr': optimizer1.state_dict()['param_groups'][0]['lr'],
                                 'loss': loss.item()})
                            pbar.update()
                        pbar.set_postfix(
                            {'stage': 'train', 'lr': optimizer1.state_dict()['param_groups'][0]['lr'],
                             'loss': all_loss / len(train_data)})
                        pbar.update()

                if True:
                    net.eval()
                    accuracy = 0
                    with torch.no_grad():
                        with tqdm(total=math.ceil(len(val_data) / 1),
                                  desc=f'Epoch{tau}/{args.FCSSN_epoch}', postfix=dict, mininterval=0.3) as pbar:
                            for sequence, (the_val_data, the_val_label) in enumerate(val_loader):
                                out_result = net(the_val_data.to(device))

                                out_result = F.softmax(out_result, dim=1)

                                out_result = out_result.cpu().detach().numpy()
                                out_result = out_result.reshape((NUM_CLASSES, input_size[0], input_size[1]))

                                out_result = np.argmax(out_result, 0) * 255
                                out_result = out_result.ravel()

                                the_val_label = the_val_label.detach().numpy()
                                the_val_label = the_val_label.reshape((256, 256))
                                the_val_label = the_val_label.ravel()
                                cm = confusion_matrix(the_val_label, out_result)

                                if cm.shape == (1, 1):
                                    this_sequence_accuracy = 0.5
                                else:
                                    this_sequence_accuracy = (cm[0][0] + cm[1][1]) / the_val_label.shape[0]
                                accuracy = accuracy + this_sequence_accuracy

                                pbar.set_postfix(
                                    {'stage': 'val', 'accuracy': this_sequence_accuracy})
                                pbar.update()

                            pbar.set_postfix(
                                {'stage': 'val', 'accuracy': accuracy / len(val_data)})
                            pbar.update()

                if tau == (args.FCSSN_epoch - 1):
                    files1 = os.listdir(args.inputDataPath).copy()
                    for ii in range(len(files1)):
                        namep1 = str(files1[ii])[:-4]
                        image = Image.open(input_path + str(files1[ii]))
                        image = image.resize((input_size[0], input_size[1]))
                        image = np.array(image)

                        if image.ndim == 2:
                            image = np.array([image])
                        elif image.ndim >= 3:
                            image = np.transpose(image, [2, 0, 1])
                        else:
                            raise SystemExit('please enter correct size, e.g.(width,height)、(width,height,channel)')

                        data2 = torch.from_numpy(np.array([image.astype('float32') / 255.])).to(device)

                        final_out_result = net(data2)
                        final_out_result = F.softmax(final_out_result, dim=1)

                        if use_gpu:
                            final_out_result = final_out_result.cpu().detach().numpy()
                        else:
                            final_out_result = final_out_result.detach().numpy()
                        final_out_result = final_out_result.reshape((NUM_CLASSES, input_size[0], input_size[1]))

                        final_out_result = np.argmax(final_out_result, 0) * 255
                        cv2.imwrite(args.outputDataPath + '/%sFCSSNout' % idudl_round + '\%s.png' % namep1,
                                    final_out_result)

                    save_name = "Epoch" + str(tau) + "-train_loss-%.6f" % (
                            all_loss / len(train_data)) + "-val_accuracy-%.6f" % (
                                        accuracy / len(val_data)) + ".pth"
                    save_path = args.outputDataPath + '/%sFCSSNpth/' % idudl_round + save_name
                    torch.save(net.state_dict(), save_path)
                    scheduler.step()
                    # writetxt(tau, all_loss / len(train_data), accuracy / len(val_data))

            break



main(args.inputDataPath)