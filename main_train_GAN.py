# ------------------------------
# Residual Dense Network
# ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim
import math
from math import log10

from model import sgsrnet
from SalUNet import unet_new
from data import *
from utils import *
import time
import LPIPS_models



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3" # "3, 4, 5"

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
parser.add_argument('--results', default='./result/newtrain', help='datasave directory')

parser.add_argument('--G_model', default='SGSR', help='G_model')
parser.add_argument('--D_model', default='RelaD', help='D_model')

parser.add_argument('--need_patch', default=False, help='get patch form image')
parser.add_argument('--RandInit', default=False, help='Rand Initialization')

# network parameters
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=96, help='patch size')  # output patch size

# training parameters
parser.add_argument('--nThreads', type=int, default=24, help='number of threads for data loading')
parser.add_argument('--train_extract_patches', type=int, default=1, help='extracted training patch numbers in one frame')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size for training') # 16
parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate')
parser.add_argument('--lr_D', type=float, default=1e-6, help='Discriminator learning rate')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--lrDecay', type=int, default=5, help='the epochs of half lr')
parser.add_argument('--decayType', default='step', help='decay type')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')

parser.add_argument('--finetuned', default=False, help='fintuning')
parser.add_argument('--finetune_G_path', default='', help='start epoch') # best PSNR
parser.add_argument('--finetune_D_path', default='None', help='start epoch')
parser.add_argument('--pretrained_sal_path', default='./SalUNet/UNet_atten_new/best-model_epoch-389_mae-0.1335_loss-0.3685.pth', help='pretrained pfan path')

parser.add_argument('--lambda_adv', type=float, default=0.1, help='lambda for adverserial loss')
parser.add_argument('--lambda_pixel', type=float, default=0.5, help='lambda for pixel loss')
parser.add_argument('--lambda_vgg', type=float, default=0.5, help='perceptual loss weights')
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
        # init.kaiming_normal(m.weight.data)
        #init.xavier_normal_(m.weight.data)
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

            imgSEG_lr = F.interpolate(imgSEG, scale_factor=1/4, mode='bicubic', align_corners=False)
            pred_mask, img_feat = pretrained_sal(imgIN)

            imgGen = model(imgIN, img_feat)

        if num_test == 0 or num_test == 1  or num_test == 2:

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

    ######################### G, D model ######################################
    #  select network
    if args.G_model == 'SGSR':
        generator = sgsrnet.GeneratorRRDB_SGSR(channels=args.nChannel, filters=args.nFeat, num_res_blocks=args.nBlock, num_upsample=int(args.scale/2))
        pretrained_sal = unet_new.UNetAttenModel()

    hr_shape = (args.patchSize, args.patchSize)
    if args.D_model == 'RelaD':
        discriminator = sgsrnet.Discriminator(input_shape=(args.nChannel, *hr_shape))

    ################# load pretrained SAL UNet weights ##############################
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

    if args.start_epoch == 0:
        if args.finetuned == True:
            gen_chkpt = torch.load("%s" % (args.finetune_G_path), map_location=torch.device("cpu"))
            state_dict = OrderedDict()
            for k, v in gen_chkpt.items():
                name = k[7:] # remove `module.`
                state_dict[name] = v
            generator.load_state_dict(state_dict)
            print("## Load Generator weights from ", args.finetune_G_path)
            
            torch.save(discriminator.state_dict(), "%s/discriminator_init.pt" % (args.results))
            print("## Initialize Discriminator weights ", "%s/discriminator_init.pt" % (args.results))

    else:
        gen_chkpt = torch.load("%s/generator_%d.pt" % (args.results, args.start_epoch-1), map_location=torch.device("cpu"))
        state_dict = OrderedDict()
        for k, v in gen_chkpt.items():
            name = k[7:] # remove `module.`
            state_dict[name] = v
        generator.load_state_dict(state_dict)
        print("## Load Generator weights from ", "%s/generator_%d.pt" % (args.results, args.start_epoch-1))
        
        dis_chkpt = torch.load("%s/discriminator_%d.pt" % (args.results, args.start_epoch-1), map_location=torch.device("cpu"))
        state_dict = OrderedDict()
        for k, v in dis_chkpt.items():
            name = k[7:] # remove `module.`
            state_dict[name] = v
        discriminator.load_state_dict(state_dict)
        print("## Load Discriminator weights from ", "%s/discriminator_%d.pt" % (args.results, args.start_epoch-1))

        ################################# FOR SINGLE GPU ####################################
        # generator.load_state_dict(torch.load("%s/generator_%d.pt" % (args.results, args.start_epoch-1)))
        #discriminator.load_state_dict(torch.load("%s/discriminator_%d.pt" % (args.results, args.start_epoch-1)))
        #discriminator_vgg.load_state_dict(torch.load("%s/discriminator_vgg_%d.pt" % (args.results, args.start_epoch-1)))
        #####################################################################################

    # generator.apply(weights_init)
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

    ########################### D model ###################################
    discriminator = torch.nn.DataParallel(discriminator)
    discriminator.cuda()
    discriminator.train()

    D_Numparams = count_parameters(discriminator)
    sys.stdout.flush()
    logFile.write('D_img_params: ' + str(D_Numparams) + '\n')
    logFile.flush()
    print('D_img_num: ' + str(D_Numparams))



    ######################## VGG 19 model ################################
    vgg19_extractor = sgsrnet.FeatureExtractor()
    vgg19_extractor = torch.nn.DataParallel(vgg19_extractor)
    vgg19_extractor.cuda()
    vgg19_extractor.eval()

    ######################## LPIPS model ################################
    # LPIPS_loss_fn = LPIPS_models.PerceptualLoss(use_gpu=True, gpu_ids=[0, 1, 2], version='0.1')
    # LPIPS_loss_fn = torch.nn.DataParallel(LPIPS_loss_fn)
    

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_D, betas=(0.9, 0.999))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # load data
    dataloader = get_dataset(args)
    testdataloader = get_testdataset(args)

    # loss
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)
    criterion_vgg = torch.nn.L1Loss().to(device)

    avg_psnr = test(args, generator, pretrained_sal, testdataloader, 0)

    total_loss = 0
    total_time = 0
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        learning_rate_G = set_lr(args, epoch, optimizer_G, args.lr_G)
        learning_rate_D = set_lr(args, epoch, optimizer_D, args.lr_D)

        for batch, (imgGT, imgIN, imgSEG) in enumerate(dataloader): # add imgSEG

            batches_done = epoch * len(dataloader) + batch
            ## GT
            imgGT = Variable(imgGT.cuda())
            ## Input
            imgIN = Variable(imgIN.cuda())
            # imgIN_lr = F.interpolate(imgIN, scale_factor=4, mode='bicubic', align_corners=False)
            imgSEG = Variable(imgSEG.cuda())

            norm_t = torch.ones(args.batchSize)*imgGT.size(1)*imgGT.size(2)*imgGT.size(3)
            norm_t = Variable(norm_t.cuda())

            

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((args.batchSize, 3, 24, 24))).cuda())
            fake = Variable(Tensor(np.zeros((args.batchSize, 3, 24, 24))).cuda())

            # ================
            # train generator
            # ================

            optimizer_G.zero_grad()
            # imgGen:  torch.Size([2, 3, 384, 384]) , imgGT:  torch.Size([2, 3, 384, 384]) , imgSEG:  torch.Size([2, 384, 384])

            pred_mask, img_feat = pretrained_sal(imgIN)
            imgGen = generator(imgIN, img_feat)


            # Pixel loss (L1 loss)
            loss_pixel = criterion_pixel(imgGen, imgGT)
            imgSEGsum = torch.sum(torch.sum(torch.sum(imgSEG, dim=1), dim=1), dim=1)
            norm_t = torch.div(norm_t, imgSEGsum)
            norm_t = norm_t.repeat_interleave(imgGT.size(1)*imgGT.size(2)*imgGT.size(3))
            norm_t = torch.reshape(norm_t, (args.batchSize, imgGT.size(1), imgGT.size(2), imgGT.size(3)))
            loss_obj = criterion_pixel(imgGen*imgSEG*norm_t, imgGT*imgSEG*norm_t)
            loss_L1 = args.lambda_pixel * loss_pixel + args.lambda_obj * loss_obj

            mask_all = (imgSEG > 0).float()
            # anti_mask_all = 1 - mask_all


            Mreal = imgGT
            Mfake = imgGen


            pred_real = discriminator(Mreal*mask_all)
            pred_fake = discriminator(Mfake*mask_all)

            Mreal_feat = vgg19_extractor(Mreal) # torch.Size([2, 512, 24, 24])
            Mfake_feat = vgg19_extractor(Mfake)
            loss_VGG = criterion_vgg(Mfake_feat, Mreal_feat)

            # Adversarial loss (relativistic average GAN)
            loss_GAN_img = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            loss_GAN = loss_GAN_img

            
            """
            # LPIPS loss
            LPIPS_loss_pixel = LPIPS_loss_fn.forward(imgGen, imgGT, normalize=False)
            loss_LPIPS_pixel = LPIPS_loss_pixel.mean()
            LPIPS_loss_obj = LPIPS_loss_fn.forward(imgGen*imgSEG*norm_t, imgGT*imgSEG*norm_t, normalize=False)
            loss_LPIPS_obj = LPIPS_loss_obj.mean()
            loss_LPIPS = args.lambda_pixel * loss_LPIPS_pixel + args.lambda_obj * loss_LPIPS_obj
            #loss_LPIPS = LPIPS_loss.mean()
            """

            # Total generator loss
            loss_G = loss_L1 + args.lambda_adv * loss_GAN + args.lambda_vgg * loss_VGG
            loss_G.backward()
            optimizer_G.step()

            # ================
            # train discriminator
            # ================

            optimizer_D.zero_grad()
            
            pred_real = discriminator((Mreal*mask_all).detach()) # print("pred_real: ", pred_real.size()) # torch.Size([2, 3, 24, 24])
            pred_fake = discriminator((Mfake*mask_all).detach())
            
            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total D loss
            if args.D_model == 'RelaD':
                loss_D = (loss_real + loss_fake) / 2


            loss_D.backward()
            optimizer_D.step()

            if batches_done % args.sample_interval == 0:

                avg_psnr = test(args, generator, pretrained_sal, testdataloader, batches_done)

                end = time.time()
                test_time = end - start
                start = time.time()

                # --------------
                #  Log Progress
                # --------------
                if args.D_model == 'RelaD':
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, vgg: %f, adv: %f, L1: %f (%f, %f)] [Val PSNR-Y: %f] [Time: %f]"
                        #"[Epoch %d/%d] [Batch %d/%d] [D loss: %f (%f, %f)] [G loss: %f, adv: %f, L1: %f (%f, %f)] [Val PSNR-Y: %f] [Time: %f]"
                        % (
                            epoch, args.epochs,
                            batch, len(dataloader),
                            loss_D.item(),
                            #loss_D_img.item(), loss_D_feat.item(),
                            loss_G.item(),
                            loss_VGG.item(),
                            # loss_LPIPS.item(),
                            loss_GAN.item(), # loss_GAN_img.item(), loss_GAN_feat.item(),
                            loss_L1.item(), loss_pixel.item(), loss_obj.item(),
                            avg_psnr,
                            test_time
                        )
                    )
                    sys.stdout.flush()

                    logFile.write(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, vgg: %f, adv: %f, L1: %f (%f, %f)] [Val PSNR-Y: %f] [Time: %f]\n"
                        #"[Epoch %d/%d] [Batch %d/%d] [D loss: %f (%f, %f)] [G loss: %f, adv: %f, L1: %f (%f, %f)] [Val PSNR-Y: %f] [Time: %f]\n"
                        % (
                            epoch, args.epochs,
                            batch, len(dataloader),
                            loss_D.item(), 
                            #loss_D_img.item(), loss_D_feat.item(),
                            loss_G.item(),
                            loss_VGG.item(),
                            # loss_LPIPS.item(),
                            loss_GAN.item(), # loss_GAN_img.item(), loss_GAN_feat.item(),
                            loss_L1.item(), loss_pixel.item(), loss_obj.item(),
                            avg_psnr,
                            test_time
                        ))
                    logFile.flush()
                
            if batches_done % args.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(), "%s/generator_%d_%d.pt" % (args.results, epoch, batches_done))
                torch.save(discriminator.state_dict(), "%s/discriminator_%d_%d.pt" % (args.results, epoch, batches_done))

            if (batches_done+1) % len(dataloader) == 0:
                torch.save(generator.state_dict(), "%s/generator_%d.pt" % (args.results, epoch))
                torch.save(discriminator.state_dict(), "%s/discriminator_%d.pt" % (args.results, epoch))
                
if __name__ == '__main__':
    train(args)
