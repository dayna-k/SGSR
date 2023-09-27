# ------------------------------
# Residual Dense Network
# ------------------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim
import math
from math import log10
from torchvision.utils import save_image, make_grid
from collections import OrderedDict
# from torch.utils.tensorboard import SummaryWriter

from model import esrgan, sgsrnet
from SalUNet import unet_new
from data import *
from utils import *
import time
from scipy import io
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import LPIPS_models


parser = argparse.ArgumentParser(description='Super Resolution')

# validation data
parser.add_argument('--val_GTdataDir', default='../../Dataset/COCO/val2017_crop_new', help='GT dataset directory')
parser.add_argument('--val_INdataDir', default='../../Dataset/COCO/val2017_crop_LR_x4_new', help='LR dataset directory')
parser.add_argument('--val_SEGdataDir', default='../../Dataset/COCO/val2017_crop_mask/mask_score_norm', help='sal score map dataset directory')

parser.add_argument('--results_folder', default='result/out/test59_atten', help='out folder directory')
parser.add_argument('--pretrained_model', default='result/ckpt/generator_59.pt', help='save result')
parser.add_argument('--pretrained_sal_path', default='./SalUNet/UNet_atten_new/best-model_epoch-389_mae-0.1335_loss-0.3685.pth', help='pretrained pfan path')

parser.add_argument('--G_model', default='SGSR', help='G_model')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')

parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--scale', type=float, default=4, help='scale output size /input size')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

args = parser.parse_args()

def save_arg(args):
    if os.path.exists(args.results_folder + '/config.txt'):
        logFile = open(args.results_folder + '/config.txt', 'a')
        for arg in vars(args):
            logFile.write('{}: {}\n'.format(arg, getattr(args, arg)))
        logFile.write('\n')
        logFile.flush()

    else:
        logFile = open(args.results_folder + '/config.txt', 'w')
        for arg in vars(args):
            logFile.write('{}: {}\n'.format(arg, getattr(args, arg)))
        logFile.write('\n')
        logFile.flush()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        # init.kaiming_normal(m.weight.data)
        init.xavier_normal_(m.weight.data)

def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        # print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True

def get_testdataset(args):
    data_test = SR_test(args)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1,
                                             drop_last=True, shuffle=False, num_workers=int(args.nThreads),
                                             pin_memory=False)
    return dataloader


def Denormalization(input):
    output = input.cpu()
    output = output.data.squeeze(0)

    # denormalization
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for t, m, s in zip(output, mean, std):
        t.mul_(s).add_(m)
    output = output.numpy()
    output *= 255.0
    output = output.clip(0, 255)
    output = np.uint8(output)
    output = np.float32(output)
    ts = (1, 2, 0)
    output = output.transpose(ts)
    return output


def Normalize_LPIPIS(input):
    output = input.cpu()
    output = output.data.squeeze(0)

    # denormalization
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for t, m, s in zip(output, mean, std):
        t.mul_(s).add_(m)

    output = torch.clamp(output, 0, 1)
    output = output.data.unsqueeze(0)
    return output


def mask_trans(input):
    output = input.cpu()
    output = output.data.squeeze(0)
    output = output.numpy()
    output = np.uint8(output)
    output = np.float32(output)
    ts = (1, 2, 0)
    output = output.transpose(ts)
    return output


def ycbcr2rgb(ycbcr):
    R = torch.clamp(ycbcr[:,0,:,:] * 1.1644 + ycbcr[:,2,:,:] * 1.5960 - 222.921, min=0, max=255)
    G = torch.clamp(ycbcr[:,0,:,:] * 1.1644 + ycbcr[:,1,:,:] * (-0.3918) + ycbcr[:,2,:,:] * (-0.8130) - (-135.5835), min=0, max=255)
    B = torch.clamp(ycbcr[:,0,:,:] * 1.1644 + ycbcr[:,1,:,:] * (2.0172) - (276.828), min=0, max=255)
    return torch.cat((R.unsqueeze(dim=1), G.unsqueeze(dim=1), B.unsqueeze(dim=1)), dim=1)


def rgb2y(rgb):
    # input shape: (384, 384, 3)
    Y = rgb[:,:,0] * 0.2568 + rgb[:,:,1] * 0.5041 + rgb[:,:,2] * 0.0979 + 16
    Y = Y.clip(0, 255)
    return Y



def test(args):
    # SR network
    create_exp_dir(args.results_folder)
    #create_exp_dir(args.results_folder+"/gen")
    #create_exp_dir(args.results_folder+"/PFAN_mask")
    #create_exp_dir(args.results_folder+"/PFAN_feat")
    #create_exp_dir(args.results_folder+"/Weighted_Atten")
    #create_exp_dir(args.results_folder+"/Img_Atten")
    #create_exp_dir(args.results_folder+"/Important_Atten")
    #create_exp_dir(args.results_folder+"/In_feat")
    #create_exp_dir(args.results_folder+"/Out_feat")

    # create log file
    if os.path.exists(args.results_folder + '/log_new.txt'):
        logFile = open(args.results_folder + '/log_new.txt', 'a')
    else:
        logFile = open(args.results_folder + '/log_new.txt', 'w')
    # create config file
    save_arg(args)

    #  select network
    if args.G_model == 'SGSR':
        #generator_base = esrgan.GeneratorRRDB(channels=args.nChannel, filters=args.nFeat, num_res_blocks=args.nBlock, num_upsample=int(args.scale/2))
        generator = sgsrnet.GeneratorRRDB_PFAN(channels=args.nChannel, filters=args.nFeat, num_res_blocks=args.nBlock, num_upsample=int(args.scale/2))
        pretrained_sal = unet_new.UNetAttenModel() 
    
    ################# load pretrained PFAN weights ##############################
    unet_chkpt = torch.load(args.pretrained_sal_path, map_location=torch.device("cpu"))
    pretrained_sal.load_state_dict(unet_chkpt['model'])
    # pretrained_pfan = torch.nn.DataParallel(pretrained_pfan)
    pretrained_sal.to("cuda")
    pretrained_sal.eval()
    UNet_Numparams = count_parameters(pretrained_sal)
    sys.stdout.flush()
    logFile.write('SAL_params: ' + str(UNet_Numparams) + '\n')
    logFile.flush()
    print('SAL_num: ' + str(UNet_Numparams))
    ##############################################################################
    
    # generator.apply(weights_init)
    
    # generator.load_state_dict(torch.load(args.pretrained_model))
    
    state_dict = torch.load(args.pretrained_model, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    generator.load_state_dict(new_state_dict)
    generator.cuda(device=args.gpu)
    generator.eval()
    


    G_Numparams = count_parameters(generator)
    sys.stdout.flush()
    logFile.write('G_params: ' + str(G_Numparams) + '\n')
    logFile.flush()
    print('G_num: ' + str(G_Numparams))
    
    testdataloader = get_testdataset(args)
    loss_fn_alex = LPIPS_models.LPIPS(net='alex')

    avg_psnr = 0
    avg_psnr_mask_0 = 0
    avg_psnr_mask_1 = 0
    avg_psnr_anti_mask_0 = 0
    avg_psnr_anti_mask_1 = 0

    avg_ssim = 0
    avg_ssim_mask_0 = 0
    avg_ssim_mask_1 = 0
    avg_ssim_anti_mask_0 = 0
    avg_ssim_anti_mask_1 = 0

    avg_lpips = 0
    avg_lpips_mask_1 = 0

    num_test = 0

    for batch, (imgGT, imgIN, imgSEG) in enumerate(testdataloader):
        with torch.no_grad():

            ## GT (3ch - Y Cb Cr)
            imgGT = Variable(imgGT.cuda(device=args.gpu))

            ## Input
            imgIN = Variable(imgIN.cuda(device=args.gpu))
            imgSEG = Variable(imgSEG.cuda(device=args.gpu))

            #img_feat = pretrained_pfan(imgIN)
            pred_mask, img_feat = pretrained_sal(imgIN)
            # imgSEG_lr = F.interpolate(imgSEG, scale_factor=1 / 4, mode='bicubic', align_corners=False)
            # imgGen = generator(imgIN)
            imgGen = generator(imgIN, img_feat)
            # imgGen, w_sa, sa, sa_mask, in_feat, out_feat = generator(imgIN, img_feat)


            imgMASK0 = (imgSEG > 0).float()
            imgMASK1 = (imgSEG == 1).float()
       
        if num_test >= 0:
            imgBIC = F.interpolate(imgIN, scale_factor=args.scale, mode='bicubic', align_corners=False)
            #imgBIC = F.upsample(imgIN.cuda(device=args.gpu), scale_factor=args.scale, mode='bicubic', align_corners=False)
            """
            #img_grid = torch.cat((imgBIC, imgGen, imgGT), -1)
            #img_grid = torch.cat((imgBIC, imgGen, imgGT), -1)
            img_grid = imgGen
            img_grid = img_grid.cpu()
            img_grid = img_grid.data.squeeze(0)
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            for t, m, s in zip(img_grid, mean, std):
                t.mul_(s).add_(m)

            #img_grid = torch.cat((img_grid, imgSEG.cpu().squeeze(0), imgMASK1.cpu().squeeze(0)), -1)
            img_grid = img_grid.numpy() # (3, 384, 1920)
            img_grid = img_grid * 255.0
            img_grid = img_grid.clip(0, 255)

            output_rgb = np.zeros((np.shape(img_grid)[1], np.shape(img_grid)[2], np.shape(img_grid)[0])) # (384, 1920, 3) BGR
            output_rgb[:, :, 0] = img_grid[2]
            output_rgb[:, :, 1] = img_grid[1]
            output_rgb[:, :, 2] = img_grid[0]

            output = Image.fromarray(np.uint8(output_rgb), mode='RGB')
            # print(type(output_rgb))
            # sr_img = cv2.imread("%s/%d_sr.png" % (args.results_folder, num_test))
            # sr_img = cv2.medianBlur(sr_img, 3)
            # cv2.imwrite("%s/%d_sr_blur3.png" % (args.results_folder, num_test), sr_img)
            output.save("%s/%d.png" % (args.results_folder, num_test))
            """ 
            """
            ########### save features
            # pred_mask (1, 96, 96)
            #print("pred_mask size: ", pred_mask.size())
            #pred_mask = np.squeeze(pred_mask.cpu().numpy(), axis=(0, 1))
            #cv2.imwrite("%s/PFAN_mask/%d.png" % (args.results_folder, num_test), pred_mask*255)

            # img_feat (128, 96, 96)
            #print("img_feat size: ", img_feat.size())
            #img_feat = img_feat.cpu().data.squeeze(0)
            #img_feat = img_feat.numpy()
            #for i in range (np.shape(img_feat)[0]):
            #    cv2.imwrite("%s/PFAN_feat/%d_%d.png" % (args.results_folder, num_test, i), img_feat[i]*255)

            # w_sa (1, 9216, 9216)
            #print("weigted_feat size: ", w_sa.size())
            w_sa = w_sa.cpu().data.squeeze(0)
            w_sa = w_sa.numpy()
            #w_sa *= (255.0/w_sa.max())
            #print(np.max(w_sa), np.min(w_sa))
            i_max, i_min = np.max(w_sa), np.min(w_sa)
            #i_min = np.min(w_sa)
            w_sa -= i_min
            w_sa *= (255.0/(i_max - i_min))
            cv2.imwrite("%s/Weighted_Atten/%d.png" % (args.results_folder, num_test), w_sa)

            # sa (1, 9216, 9216)
            #print("img_atten size: ", sa.size())
            sa = sa.cpu().data.squeeze(0)
            sa = sa.numpy()
            i_max, i_min = np.max(sa), np.min(sa)
            sa -= i_min
            sa *= (255.0/(i_max - i_min))
            #print(np.max(sa), np.min(sa))
            cv2.imwrite("%s/Img_Atten/%d.png" % (args.results_folder, num_test), sa)

            # sa_mask (1, 9216, 9216)
            #print("important_atten size: ", sa_mask.size())
            sa_mask = sa_mask.cpu().data.squeeze(0)
            sa_mask = sa_mask.numpy()
            i_max, i_min = np.max(sa_mask), np.min(sa_mask)
            sa_mask -= i_min
            sa_mask *= (255.0/(i_max - i_min))
            # sa_mask *= (255.0/sa_mask.max())
            #print(np.max(sa_mask), np.min(sa_mask))
            cv2.imwrite("%s/Important_Atten/%d.png" % (args.results_folder, num_test), sa_mask)
            
            # in_feat (64, 96, 96)
            #print("in_feat size: ", in_feat.size())
            in_feat = in_feat.cpu().data.squeeze(0)
            in_feat = in_feat.numpy()
            for i in range (np.shape(in_feat)[0]):
                cv2.imwrite("%s/In_feat/%d_%d.png" % (args.results_folder, num_test, i), in_feat[i]*255)

            # out_feat (64, 96, 96)
            # print("out_feat size: ", out_feat.size())
            out_feat = out_feat.cpu().data.squeeze(0)
            out_feat = out_feat.numpy()
            for i in range (np.shape(out_feat)[0]):
                cv2.imwrite("%s/Out_feat/%d_%d.png" % (args.results_folder, num_test, i), out_feat[i]*255)
            """


        area_0 = float(torch.sum(imgMASK0[:, :, 8:-8, 8:-8]) / 3)
        anti_area_0 = float(368 * 368 - area_0)

        area_1 = float(torch.sum(imgMASK1[:, :, 8:-8, 8:-8]) / 3)
        anti_area_1 = float(368 * 368 - area_1)


        # if area_1 > 384*384*0.1 and anti_area_1 > 384*384*0.1 and area_0 > 384*384*0.1 and anti_area_0 > 384*384*0.1:
        if area_1 > 368*368*0.1 and anti_area_0 > 368*368*0.1: #384*384-368*368 = 12032
            gen = Denormalization(imgGen)
            # print(gen.shape)
            gen = rgb2y(gen)
            # print(gen.shape)
            GT = Denormalization(imgGT)
            GT = rgb2y(GT)

            GT_norm = Normalize_LPIPIS(imgGT)
            gen_norm = Normalize_LPIPIS(imgGen)

            GT_norm_1 = Normalize_LPIPIS(imgGT*imgMASK1)
            gen_norm_1 = Normalize_LPIPIS(imgGen*imgMASK1)

            mask_0 = mask_trans(imgMASK0)
            gen_mask_0 = gen * mask_0[:, :, 0]
            GT_mask_0 = GT * mask_0[:, :, 0]

            gen_anti_mask_0 = gen - gen_mask_0
            GT_anti_mask_0 = GT - GT_mask_0

            mask_1 = mask_trans(imgMASK1)
            gen_mask_1 = gen * mask_1[:, :, 0]
            GT_mask_1 = GT * mask_1[:, :, 0]

            gen_anti_mask_1 = gen - gen_mask_1
            GT_anti_mask_1 = GT - GT_mask_1

            mse = ((GT[8:-8, 8:-8] - gen[8:-8, 8:-8]) ** 2).sum() / float(368 * 368)
            psnr = 10 * log10(255 * 255 / (mse + 10 ** (-10)))
            ssim = compare_ssim(GT[8:-8, 8:-8], gen[8:-8, 8:-8], data_range = gen[8:-8, 8:-8].max() - gen[8:-8, 8:-8].min())
            lpips = loss_fn_alex(GT_norm.cpu(), gen_norm.cpu(), normalize=True)

            mse_mask_0 = ((gen_mask_0[8:-8, 8:-8] - GT_mask_0[8:-8, 8:-8]) ** 2).sum() / area_0
            psnr_mask_0 = 10 * log10(255 * 255 / (mse_mask_0 + 10 ** (-10)))
            ssim_mask_0 = compare_ssim(GT_mask_0[8:-8, 8:-8], gen_mask_0[8:-8, 8:-8], data_range = gen_mask_0[8:-8, 8:-8].max() - gen_mask_0[8:-8, 8:-8].min())

            mse_anti_mask_0 = ((gen_anti_mask_0[8:-8, 8:-8] - GT_anti_mask_0[8:-8, 8:-8]) ** 2).sum() / anti_area_0
            psnr_anti_mask_0 = 10 * log10(255 * 255 / (mse_anti_mask_0 + 10 ** (-10)))
            ssim_anti_mask_0 = compare_ssim(GT_anti_mask_0[8:-8, 8:-8], gen_anti_mask_0[8:-8, 8:-8], data_range = gen_anti_mask_0[8:-8, 8:-8].max() - gen_anti_mask_0[8:-8, 8:-8].min())

            mse_mask_1 = ((gen_mask_1[8:-8, 8:-8] - GT_mask_1[8:-8, 8:-8]) ** 2).sum() / area_1
            psnr_mask_1 = 10 * log10(255 * 255 / (mse_mask_1 + 10 ** (-10)))
            ssim_mask_1 = compare_ssim(GT_mask_1[8:-8, 8:-8], gen_mask_1[8:-8, 8:-8], data_range = gen_mask_1[8:-8, 8:-8].max() - gen_mask_1[8:-8, 8:-8].min())
            lpips_mask_1 = loss_fn_alex(GT_norm_1.cpu(), gen_norm_1.cpu(), normalize=True)
            lpips_mask_1 = lpips_mask_1 * (384*384) / float(torch.sum(imgMASK1) / 3)

            mse_anti_mask_1 = ((gen_anti_mask_1[8:-8, 8:-8] - GT_anti_mask_1[8:-8, 8:-8]) ** 2).sum() / anti_area_1
            psnr_anti_mask_1 = 10 * log10(255 * 255 / (mse_anti_mask_1 + 10 ** (-10)))
            ssim_anti_mask_1 = compare_ssim(GT_anti_mask_1[8:-8, 8:-8], gen_anti_mask_1[8:-8, 8:-8], data_range = gen_anti_mask_0[8:-8, 8:-8].max() - gen_anti_mask_0[8:-8, 8:-8].min())


            avg_psnr += psnr
            avg_psnr_mask_0 += psnr_mask_0
            avg_psnr_mask_1 += psnr_mask_1
            avg_psnr_anti_mask_0 += psnr_anti_mask_0
            avg_psnr_anti_mask_1 += psnr_anti_mask_1

            avg_ssim += ssim
            avg_ssim_mask_0 += ssim_mask_0
            avg_ssim_mask_1 += ssim_mask_1
            avg_ssim_anti_mask_0 += ssim_anti_mask_0
            avg_ssim_anti_mask_1 += ssim_anti_mask_1

            avg_lpips += lpips
            avg_lpips_mask_1 += lpips_mask_1

            num_test = num_test + 1

            sys.stdout.flush()
            logFile.write("[Test %d] [SSIM: %f, PSNR: %f, LPIPS: %f][(Mask0) PSNR: %f, %f, SSIM: %f, %f, mask area: %d][(Mask1) PSNR: %f, %f, SSIM: %f, %f, LPIPS: %f, mask area: %d] \n"
                          % (num_test, ssim, psnr, lpips, psnr_mask_0, psnr_anti_mask_0, ssim_mask_0, ssim_anti_mask_0, area_0, psnr_mask_1, psnr_anti_mask_1, ssim_mask_1, ssim_anti_mask_1, lpips_mask_1, area_1))
            logFile.flush()
            print("[Test %d] [SSIM: %f, PSNR: %f LPIPS: %f][(Mask0) PSNR: %f, %f, SSIM: %f, %f, mask area: %d][(Mask1) PSNR: %f, %f, SSIM: %f, %f, LPIPS: %f, mask area: %d]"
                  % (num_test, ssim, psnr, lpips, psnr_mask_0, psnr_anti_mask_0, ssim_mask_0, ssim_anti_mask_0, area_0, psnr_mask_1, psnr_anti_mask_1, ssim_mask_1, ssim_anti_mask_1, lpips_mask_1, area_1))


    sys.stdout.flush()
    logFile.write("[Test for %d images] [AVG-SSIM: %f, AVG-PSNR: %f, AVG-LPIPS: %f][(Mask0) PSNR: %f, %f, SSIM: %f, %f][(Mask1) PSNR: %f, %f, SSIM: %f, %f, LPIPS: %f] \n"
                  % (num_test, avg_ssim / num_test, avg_psnr / num_test, avg_lpips / num_test, avg_psnr_mask_0 / num_test, avg_psnr_anti_mask_0 / num_test, avg_ssim_mask_0 / num_test, avg_ssim_anti_mask_0 / num_test, avg_psnr_mask_1 / num_test, avg_psnr_anti_mask_1 / num_test, avg_ssim_mask_1 / num_test, avg_ssim_anti_mask_1 / num_test, avg_lpips_mask_1 / num_test))
    logFile.flush()
    print("[Test for %d images] [AVG-SSIM: %f, AVG-PSNR: %f, AVG-LPIPS: %f][(Mask0) PSNR: %f, %f, SSIM: %f, %f][(Mask1) PSNR: %f, %f, SSIM: %f, %f, LPIPS: %f] \n"
                  % (num_test, avg_ssim / num_test, avg_psnr / num_test, avg_lpips / num_test, avg_psnr_mask_0 / num_test, avg_psnr_anti_mask_0 / num_test, avg_ssim_mask_0 / num_test, avg_ssim_anti_mask_0 / num_test, avg_psnr_mask_1 / num_test, avg_psnr_anti_mask_1 / num_test, avg_ssim_mask_1 / num_test, avg_ssim_anti_mask_1 / num_test, avg_lpips_mask_1 / num_test))


if __name__ == '__main__':
    test(args)
