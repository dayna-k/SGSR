# ------------------------------
# Residual Dense Network
# ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F # from parallel import DataParallelModel, DataParallelCriterion
from collections import OrderedDict

from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
#from torchsummary import summary
import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim
import math
from math import log10


from model import esrgan, sgsrnet
from SalUNet import unet_new
from data import *
from utils import *
import time
# import LPIPS_models

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3" # "4, 5, 6"


parser = argparse.ArgumentParser(description='super-resolution')

# train data
parser.add_argument('--GTdataDir', default='../../Dataset/COCO/train2017_crop_new', help='dataset directory')
parser.add_argument('--INdataDir', default='../../Dataset/COCO/train2017_crop_LR_x4_new', help='dataset directory')
parser.add_argument('--SEGdataDir', default='../../Dataset/COCO/train2017_crop_mask/mask_score_norm', help='dataset directory')

# validation data
parser.add_argument('--val_GTdataDir', default='../../Dataset/COCO/val_small/gt', help='dataset directory')
parser.add_argument('--val_INdataDir', default='../../Dataset/COCO/val_small/lr', help='dataset directory')
parser.add_argument('--val_SEGdataDir', default='../../Dataset/COCO/val_small/mask/mask_score_norm', help='dataset directory')

# output folder
parser.add_argument('--saveDir', default='./result', help='datasave directory')
parser.add_argument('--results', default='./result/ESRGAN_UNet_0928_reduce_double', help='datasave directory')

parser.add_argument('--G_model', default='ESRGAN_UNet', help='G_model')
parser.add_argument('--D_model', default='ESRGAN', help='D_model')

parser.add_argument('--need_patch', default=False, help='get patch form image')
parser.add_argument('--RandInit', default=False, help='Rand Initialization')

# network parameters
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=96, help='patch size')  # output patch size

# training parameters
parser.add_argument('--nThreads', type=int, default=16, help='number of threads for data loading')
parser.add_argument('--train_extract_patches', type=int, default=1, help='extracted training patch numbers in one frame')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size for training') # 16
parser.add_argument('--lr_G', type=float, default=(1e-4), help='Generator learning rate')
parser.add_argument('--lr_D', type=float, default=1e-6, help='Discriminator learning rate')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--lrDecay', type=int, default=5, help='the epochs of half lr')
parser.add_argument('--decayType', default='step', help='decay type')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--start_iter', type=int, default=0, help='start iter')
parser.add_argument('--start_with_iter', default=False, help='start with iter')

parser.add_argument('--finetuned', default=False, help='fintuning')
parser.add_argument('--finetune_path', default='', help='finetune_path')
parser.add_argument('--pretrained_sal_path', default='./SalUNet/UNet_atten_new/best-model_epoch-389_mae-0.1335_loss-0.3685.pth', help='pretrained pfan path')


parser.add_argument('--lambda_adv', type=float, default=0, help='lambda for adverserial loss')
parser.add_argument('--lambda_pixel', type=float, default=0.5, help='lambda for pixel loss')
parser.add_argument('--lambda_vgg', type=float, default=0, help='perceptual loss weights')
parser.add_argument('--lambda_obj', type=float, default=0.5, help='perceptual obj loss weights')

parser.add_argument("--warmup_batches", type=int, default=0, help="number of batches with pixel-wise loss only")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=15000, help="batch interval between model checkpoints")

parser.add_argument('--scale', type=int, default=4, help='scale output size /input size')
parser.add_argument('--multi-processing', default=True, help='use multi-gpu')
parser.add_argument('--gpu', default=[1, 2, 3], help='gpu index')


args = parser.parse_args()


def save_arg(args):
    if os.path.exists(args.results + '/config.txt'):
        logFile = open(args.results + '/config.txt', 'a')
        for arg in vars(args):
            logFile.write('{}: {}\n'.format(arg, getattr(args, arg)))
        logFile.write('\n')
        logFile.flush()

    else:
        logFile = open(args.results + '/config.txt', 'w')
        for arg in vars(args):
            logFile.write('{}: {}\n'.format(arg, getattr(args, arg)))
        logFile.write('\n')
        logFile.flush()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def set_lr(args, epoch, optimizer, start_lr):
    lrDecay = args.lrDecay
    decayType = args.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = start_lr / 2 ** epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = start_lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = start_lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_dataset(args):
    data_train = SR_train(args)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batchSize,
                                             drop_last=True, shuffle=True, num_workers=int(args.nThreads),
                                             pin_memory=False)
    return dataloader


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

def rgb2y(rgb):
    # input shape: (384, 384, 3)
    Y = rgb[:, :, 0] * 0.2568 + rgb[:, :, 1] * \
        0.5041 + rgb[:, :, 2] * 0.0979 + 16
    Y = Y.clip(0, 255)
    return Y

def test(args, model, pretrained_sal, dataloader, batches_done):
    avg_psnr = 0
    num_test = 0

    for batch, (imgGT, imgIN, imgSEG) in enumerate(dataloader):
        with torch.no_grad():

            ## GT (3ch - Y Cb Cr)
            imgGT = Variable(imgGT.cuda())
            ## Input
            imgIN = Variable(imgIN.cuda())
            # mask
            imgSEG = Variable(imgSEG.cuda())

            pred_mask, img_feat = pretrained_sal(imgIN)
            
            if imgIN.get_device() != img_feat.get_device():
                print("imgIN: ", imgIN.get_device())
                print("img_feat: ", img_feat.get_device())
            imgGen = model(imgIN, img_feat)

        if num_test == 0 or num_test == 1:

            imgBIC = F.interpolate(imgIN.cuda(), scale_factor=args.scale, mode='bicubic', align_corners=False)

            img_grid = torch.cat((imgBIC, imgGen, imgGT), -1)

            img_grid = img_grid.cpu()
            img_grid = img_grid.data.squeeze(0)
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            for t, m, s in zip(img_grid, mean, std):
                t.mul_(s).add_(m)

            img_grid = torch.cat((img_grid, imgSEG.cpu().squeeze(0)), -1)

            img_grid = img_grid.numpy()
            img_grid = img_grid*255.0
            img_grid = img_grid.clip(0, 255)

            output_rgb = np.zeros((np.shape(img_grid)[1], np.shape(img_grid)[2], np.shape(img_grid)[0]))
            output_rgb[:, :, 0] = img_grid[2]
            output_rgb[:, :, 1] = img_grid[1]
            output_rgb[:, :, 2] = img_grid[0]

            output = Image.fromarray(np.uint8(output_rgb), mode='RGB')
            output.save("%s/%d_%d.png" % (args.results, batches_done, num_test))


        gen = Denormalization(imgGen)
        GT = Denormalization(imgGT)
        gen = rgb2y(gen)
        GT = rgb2y(GT)

        mse = ((GT[8:-8, 8:-8] - gen[8:-8, 8:-8]) ** 2).mean()
        psnr = 10 * log10(255 * 255 / (mse + 10 ** (-10)))

        avg_psnr += psnr  
        num_test = num_test + 1



    return avg_psnr / num_test


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create folder
    create_exp_dir(args.results)
    # create log file
    if os.path.exists(args.results + '/log.txt'):
        logFile = open(args.results + '/log.txt', 'a')
    else:
        logFile = open(args.results + '/log.txt', 'w')
    # create config file
    save_arg(args)

    ######################### G model ######################################
    #  select network
    if args.G_model == 'SGSR':
        generator_base = esrgan.GeneratorRRDB(channels=args.nChannel, filters=args.nFeat, num_res_blocks=args.nBlock, num_upsample=int(args.scale/2))
        generator = sgsrnet.GeneratorRRDB_SGSR(channels=args.nChannel, filters=args.nFeat, num_res_blocks=args.nBlock, num_upsample=int(args.scale/2))
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

    # fine-tuning or retrain
    if args.finetuned and args.start_epoch == 0 and not args.start_with_iter:
        gen_chkpt = torch.load("%s" % (args.finetune_path), map_location=torch.device("cpu"))
        generator_base.load_state_dict(gen_chkpt)
        pretrained_dict = generator_base.state_dict()
        new_model_dict = generator.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        generator.load_state_dict(new_model_dict)

        print("## Load Generator weights from ", args.finetune_path)
        logFile.write("## Load Generator weights from %s" %(args.finetune_path))
        logFile.flush()
        torch.save(generator.state_dict(), "%s/generator_ft_init.pt" % (args.results))

        
    elif args.start_epoch != 0:
        
        if args.start_with_iter:
            gen_chkpt = torch.load("%s/generator_%d_%d.pt" % (args.results, args.start_epoch, args.start_iter), map_location=torch.device("cpu"))
            state_dict = OrderedDict()
            for k, v in gen_chkpt.items():
                name = k[7:] # remove `module.`
                state_dict[name] = v
            generator.load_state_dict(state_dict)
            print("## Load Generator weights from ", "%s/generator_%d_%d.pt" % (args.results, args.start_epoch, args.start_iter))
            logFile.write("## Load Generator weights from %s/generator_%d_%d.pt" % (args.results, args.start_epoch, args.start_iter))
            logFile.flush()
        
        else:
            gen_chkpt = torch.load("%s/generator_%d.pt" % (args.results, args.start_epoch-1), map_location=torch.device("cpu"))
            state_dict = OrderedDict()
            for k, v in gen_chkpt.items():
                name = k[7:] # remove `module.`
                state_dict[name] = v
            generator.load_state_dict(state_dict)
            print("## Load Generator weights from %s/generator_%d.pt" % (args.results, args.start_epoch-1))
            logFile.write("## Load Generator weights from %s/generator_%d.pt" % (args.results, args.start_epoch-1))
            logFile.flush()


    generator = torch.nn.DataParallel(generator) # device_ids=[3, 4, 6] -> [0, 1, 2]
    
    generator.cuda()

    print("gpu_num: ", torch.cuda.device_count())
    print("generator in: ", generator.device_ids)
    generator.train()

    G_Numparams = count_parameters(generator)
    sys.stdout.flush()
    logFile.write('G_params: ' + str(G_Numparams) + '\n')
    logFile.flush()
    print('G_num: ' + str(G_Numparams))
    # summary(generator, ((3, 384, 384), ()))

    

    ####################### Load trained Weights ###############################
    
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G, betas=(0.9, 0.999))

    # load data
    dataloader = get_dataset(args)
    testdataloader = get_testdataset(args)

    # loss
    criterion_pixel = torch.nn.L1Loss().to(device)

    avg_psnr = test(args, generator, pretrained_sal, testdataloader, 0)

    total_loss = 0
    total_time = 0
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        learning_rate_G = set_lr(args, epoch, optimizer_G, args.lr_G)
        # learning_rate_D = set_lr(args, epoch, optimizer_D, args.lr_D)


        for batch, (imgGT, imgIN, imgSEG) in enumerate(dataloader): # add imgSEG

            batches_done = epoch * len(dataloader) + batch
            if args.start_with_iter and batches_done <= args.start_iter:
                if batches_done % 5000 == 0:
                    print("PASS [epoch: %d, batch: %d/%d, batch_done: %d]" % (epoch, batch, len(dataloader), batches_done))  
                continue
            ## GT
            imgGT = Variable(imgGT.cuda())

            ## Input
            imgIN = Variable(imgIN.cuda())
            imgSEG = Variable(imgSEG.cuda())

            norm_t = torch.ones(args.batchSize)*imgGT.size(1)*imgGT.size(2)*imgGT.size(3)
            norm_t = Variable(norm_t.cuda())

            # ================
            # train generator
            # ================

            optimizer_G.zero_grad()
            pred_mask, img_feat = pretrained_sal(imgIN)


            if imgIN.get_device() != img_feat.get_device():
                print("imgIN: ", imgIN.get_device())
                print("img_feat: ", img_feat.get_device())

            imgGen = generator(imgIN, img_feat)
            
            # Pixel loss (L1 loss)
            loss_pixel = criterion_pixel(imgGen, imgGT)

            # imgGen:  torch.Size([2, 3, 384, 384]) , imgGT:  torch.Size([2, 3, 384, 384]) , imgSEG:  torch.Size([2, 384, 384])
            imgSEGsum = torch.sum(torch.sum(torch.sum(imgSEG, dim=1), dim=1), dim=1)
            norm_t = torch.div(norm_t, imgSEGsum)
            norm_t = norm_t.repeat_interleave(imgGT.size(1)*imgGT.size(2)*imgGT.size(3))
            norm_t = torch.reshape(norm_t, (args.batchSize, imgGT.size(1), imgGT.size(2), imgGT.size(3)))

            loss_obj = criterion_pixel(imgGen*imgSEG*norm_t, imgGT*imgSEG*norm_t)


            
            # Total generator loss
            loss_G = args.lambda_pixel * loss_pixel + args.lambda_obj * loss_obj
            loss_G.backward()
            optimizer_G.step()
            
            if batches_done % args.sample_interval == 0:

                avg_psnr = test(args, generator, pretrained_sal, testdataloader, batches_done)

                end = time.time()
                test_time = end - start
                start = time.time()

                # --------------
                #  Log Progress
                # --------------

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f, obj: %f] [Val PSNR-Y: %f] [Time: %f]"
                    % (
                        epoch,
                        args.epochs,
                        batch,
                        len(dataloader),
                        loss_G.item(),
                        loss_pixel.item(),
                        loss_obj.item(),
                        avg_psnr,
                        test_time
                    )
                )

                sys.stdout.flush()
                logFile.write(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f, obj: %f] [Val PSNR-Y: %f] [Time: %f] \n"
                    % (
                        epoch,
                        args.epochs,
                        batch,
                        len(dataloader),
                        loss_G.item(),
                        loss_pixel.item(),
                        loss_obj.item(),
                        avg_psnr,
                        test_time
                    ))
                logFile.flush()

            if batches_done % args.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(), "%s/generator_%d_%d.pt" % (args.results, epoch, batches_done))
            
            if (batches_done+1) % len(dataloader) == 0:
                torch.save(generator.state_dict(), "%s/generator_%d.pt" % (args.results, epoch))

if __name__ == '__main__':
    train(args)
