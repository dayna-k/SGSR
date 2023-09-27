import os
import os.path
import random
import json
import numpy as np
import cv2
import torch
import torch.utils.data as data
from PIL import Image
import natsort



def RGB_np2Tensor(imgGT, imgIN, imgSEG):
    # to Tensor
    ts = (2, 0, 1)
    # print(imgGT)
    # print(imgIN)
    imgGT = torch.Tensor(imgGT.transpose(ts).astype(float))
    imgIN = torch.Tensor(imgIN.transpose(ts).astype(float))
    imgSEG = torch.Tensor(imgSEG.transpose(ts).astype(float))

    # normaliztion [-1, 1]
    imgGT = ((imgGT / 255.0) - 0.5) * 2
    imgIN = ((imgIN / 255.0) - 0.5) * 2
    imgSEG = (imgSEG/255.0)

    return imgGT, imgIN, imgSEG

def getPatch(imgGT, imgIN, scale, patchsize):
    (ih, iw, c) = imgIN.shape
    (th, tw) = (scale * ih, scale * iw)
    tp = patchsize  # HR image patch size
    ip = tp // scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIN = imgIN[iy:iy + ip, ix:ix + ip, :]
    imgGT = imgGT[ty:ty + tp, tx:tx + tp, :]
    return imgGT, imgIN

def augment(imgGT, imgIN, imgSEG):
    if random.random() < 0.3:  # horizontal flip
        imgGT = imgGT[:, ::-1, :]
        imgIN = imgIN[:, ::-1, :]
        imgSEG = imgSEG[:, ::-1]

    if random.random() < 0.3:  # vertical flip
        imgGT = imgGT[::-1, :, :]
        imgIN = imgIN[::-1, :, :]
        imgSEG = imgSEG[::-1, :]

    rot = random.randint(0, 3)  # rotate
    imgGT = np.rot90(imgGT, rot, (0, 1))
    imgIN = np.rot90(imgIN, rot, (0, 1))
    imgSEG = np.rot90(imgSEG, rot, (0, 1))

    imgSEG_3c = np.zeros(imgGT.shape)
    imgSEG_3c[:, :, 0] = imgSEG
    imgSEG_3c[:, :, 1] = imgSEG
    imgSEG_3c[:, :, 2] = imgSEG
    return imgGT, imgIN, imgSEG_3c

class SR_test(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = args.scale
        self.channel = args.nChannel
        #self.patchsize = args.patchSize

        GTapath = args.val_GTdataDir
        INapath = args.val_INdataDir
        SEGapath = args.val_SEGdataDir

        self.dirGT = os.path.join(GTapath)
        self.dirIN = os.path.join(INapath)
        self.dirSEG = os.path.join(SEGapath)

        self.fileList = os.listdir(self.dirGT)
        # self.fileList = natsort.natsorted(self.fileList)

        self.nTrain = len(self.fileList)

    def __getitem__(self, idx):

        nameGT, nameIN, nameSEG = self.getFileName(idx)
        # print(nameGT, nameIN)
        imgIN = cv2.imread(nameIN, cv2.IMREAD_COLOR)
        imgGT = cv2.imread(nameGT, cv2.IMREAD_COLOR)

        imgSEG = cv2.imread(nameSEG, 0) # 0: read img as grayscale
        imgSEG_3c = np.zeros(imgGT.shape)
        imgSEG_3c[:, :, 0] = imgSEG
        imgSEG_3c[:, :, 1] = imgSEG
        imgSEG_3c[:, :, 2] = imgSEG

        return RGB_np2Tensor(imgGT, imgIN, imgSEG_3c)


    def __len__(self):
        return self.nTrain

    def getFileName(self, idx):
        name = self.fileList[idx]
        #nameGT = os.path.join(self.dirGT, name)
        nameGT = self.dirGT+"/"+name
        # nameIN = os.path.join(self.dirIN, name[0:-4] + 'x4.png')
        nameIN = self.dirIN + "/" + name[0:-4] + '.png'
        #nameIN = os.path.join(self.dirIN, name)
        nameSEG = self.dirSEG + "/" + name[0:-4] + '.png'

        return nameGT, nameIN, nameSEG


class SR_train(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = args.scale
        self.channel = args.nChannel
        self.patchsize = args.patchSize

        GTapath = args.GTdataDir
        INapath = args.INdataDir
        SEGapath = args.SEGdataDir

        self.dirGT = os.path.join(GTapath)
        self.dirIN = os.path.join(INapath)
        self.dirSEG = os.path.join(SEGapath)

        self.fileList = os.listdir(self.dirGT)
        self.nTrain = len(self.fileList)
        # self.json_data = self.get_json("train")

    def __getitem__(self, idx):
        #test 말고 train만
        nameGT, nameIN, nameSEG = self.getFileName(idx)
        # print("[ ",idx," ] nameGT: ", nameGT)
        name = self.fileList[idx]
        imgGT = cv2.imread(nameGT)
        imgIN = cv2.imread(nameIN)
        imgSEG = cv2.imread(nameSEG, 0)

        # nameSEG, imgSEG = self.getSEGImg(idx)
        
        real_id = int(name[:-4].lstrip('0'))

        if imgIN.any() == None:
            print("No Input Img: ", nameIN)
        if imgGT.any() == None:
            print("No GT Img: ", nameGT)
        if imgSEG.any() == None:
            print("No SEG Img: ", nameSEG)
        # imgGT, imgIN = getPatch(imgGT, imgIN, self.scale, self.patchsize)

        # get augmentation
        imgGT, imgIN, imgSEG = augment(imgGT, imgIN, imgSEG)
        #cv2.imshow("GT"+str(real_id), imgGT)
        #cv2.waitKey(1000)
        #cv2.imshow("IN"+str(real_id), imgIN)
        #cv2.waitKey(1000)
        #cv2.imshow("SEG"+str(real_id), imgSEG)
        #cv2.waitKey(1000)
        #cv2.destroyAllWindows()
        return RGB_np2Tensor(imgGT, imgIN, imgSEG)

    def __len__(self):
        return self.nTrain

    def getFileName(self, idx):
        name = self.fileList[idx]

        nameGT = self.dirGT + "/" + name
        nameIN = self.dirIN + "/" + name[0:-4] + '.png'
        nameSEG = self.dirSEG + "/" + name[0:-4] + '.png'

        return nameGT, nameIN, nameSEG

    
