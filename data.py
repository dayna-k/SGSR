import os
import os.path
import random
import json
import numpy as np
import cv2
import torch
import torch.utils.data as data
from PIL import Image


_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def image_files(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError('Image directory does not exist: {}'.format(directory))
    names = sorted(name for name in os.listdir(directory)
                   if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS
                   and os.path.isfile(os.path.join(directory, name)))
    if not names:
        raise ValueError('No supported images found in: {}'.format(directory))
    return names


def read_triplet(nameGT, nameIN, nameSEG, scale, require_score=False):
    imgGT = cv2.imread(nameGT, cv2.IMREAD_COLOR)
    imgIN = cv2.imread(nameIN, cv2.IMREAD_COLOR)
    imgSEG = cv2.imread(nameSEG, cv2.IMREAD_GRAYSCALE)
    for path, img in [(nameGT, imgGT), (nameIN, imgIN), (nameSEG, imgSEG)]:
        if img is None:
            raise ValueError('Cannot read image: {}'.format(path))
    if imgGT.shape[:2] != (imgIN.shape[0] * scale, imgIN.shape[1] * scale):
        raise ValueError('HR must be {}x LR in each dimension: {} and {}'.format(scale, nameGT, nameIN))
    if imgSEG.shape != imgGT.shape[:2]:
        raise ValueError('Score map must match HR dimensions: {}'.format(nameSEG))
    if any(size % 2 for size in imgIN.shape[:2]) or min(imgIN.shape[:2]) < 16:
        raise ValueError('LR dimensions must be even and at least 16 pixels: {}'.format(nameIN))
    if require_score and not np.any(imgSEG):
        raise ValueError('Training score map must contain nonzero values: {}'.format(nameSEG))
    return imgGT, imgIN, imgSEG


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

        self.fileList = image_files(self.dirGT)
        # self.fileList = natsort.natsorted(self.fileList)

        self.nTrain = len(self.fileList)

    def __getitem__(self, idx):

        nameGT, nameIN, nameSEG = self.getFileName(idx)
        # print(nameGT, nameIN)
        imgGT, imgIN, imgSEG = read_triplet(nameGT, nameIN, nameSEG, self.scale)
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
        nameIN = self.dirIN + "/" + os.path.splitext(name)[0] + '.png'
        #nameIN = os.path.join(self.dirIN, name)
        nameSEG = self.dirSEG + "/" + os.path.splitext(name)[0] + '.png'

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

        self.fileList = image_files(self.dirGT)
        self.nTrain = len(self.fileList)
        # self.json_data = self.get_json("train")

    def __getitem__(self, idx):
        #test 말고 train만
        nameGT, nameIN, nameSEG = self.getFileName(idx)
        # print("[ ",idx," ] nameGT: ", nameGT)
        imgGT, imgIN, imgSEG = read_triplet(
            nameGT, nameIN, nameSEG, self.scale, require_score=True)

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
        nameIN = self.dirIN + "/" + os.path.splitext(name)[0] + '.png'
        nameSEG = self.dirSEG + "/" + os.path.splitext(name)[0] + '.png'

        return nameGT, nameIN, nameSEG

    
