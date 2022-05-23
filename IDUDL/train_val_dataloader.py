import os
import cv2
import random
import shutil
from shutil import copy


def newloader_FEN(path, namep, final, round):
    if os.path.exists(path+'/%s'% round):

        cv2.imwrite(path + '/%s/' % round + '%s.png' % namep, final)
    else:
        os.makedirs(path + '/%strainlabel' % round)
        os.makedirs(path + '/%svallabel' % round)
        os.makedirs(path + '/%strainimg' % round)
        os.makedirs(path + '/%svalimg' % round)
        os.makedirs(path + '/%s' % round)
        os.makedirs(path + '/%sFCSSNpth' % round)
        os.makedirs(path + '/%sFCSSNout' % round)
        cv2.imwrite(path+'/%s/'%round + '%s.png' % namep, final)

def train_loader1(input, round):

    trainfiles = input + '%s'%round
    ai = os.listdir(trainfiles).copy()

    num_train = len(ai)

    index_list = list(range(num_train))

    random.shuffle(index_list)
    num = 0
    trainDir = input + '\\%strainlabel\\'%round

    validDir = input + '\\%svallabel\\'%round

    files1 = os.listdir(trainfiles).copy()
    for i in range(len(files1)):
        fileName = os.path.join(input+'%s\\'%round, str(files1[i]))


        if num < num_train*0.7:

            copy(fileName, trainDir + str(files1[i]))
        else:
            copy(fileName, validDir + str(files1[i]))
        num += 1

    return trainDir, validDir

def label_to_img(imgDir, trainDir, validDir):
    files1 = os.listdir(trainDir).copy()
    for i in range(len(files1)):
        img = cv2.imread(imgDir + str(files1[i]))

        cv2.imwrite(trainDir[:-11] + 'trainimg\\'+ str(files1[i]), img)

    files2 = os.listdir(validDir).copy()
    for j in range(len(files2)):
        img1 = cv2.imread(imgDir + str(files2[j]))
        cv2.imwrite(validDir[:-9] + 'valimg\\' + str(files2[j]), img1)



