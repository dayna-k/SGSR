from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#from .attention import SpatialAttention, ChannelwiseAttention

# From https://github.com/milesial/Pytorch-UNet

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ChannelAtten(nn.Module):
    """ channel atten """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_ca = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_mlp =  self.mlp(avg_ca)
        max_ca = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_mlp =  self.mlp(max_ca)
        ca_sum = avg_mlp + max_mlp
        #ca_scale = F.sigmoid(ca_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * ca_sum



class SpatialAtten(nn.Module):
    """ spatial atten """
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.saConv = nn.Conv2d(2, 1, kernel_size, padding=self.kernel_size//2, bias=False)


    def forward(self, x):
        avg_sa = torch.mean(x, dim=1, keepdim=True)
        max_sa, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.saConv(torch.cat([avg_sa, max_sa], dim=1))
        sa = torch.sigmoid(sa)
        return x * sa

class CSA(nn.Module):
    """(channel atten => spatial atten) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_atten = ChannelAtten(in_channels)
        self.spatial_atten = SpatialAtten(kernel_size=7)

        self.gamma = nn.Parameter(torch.rand(1))
    def forward(self, x):
        x_ca = self.channel_atten(x)
        x_sa = self.spatial_atten(x_ca)
        out = x_sa + x * self.gamma
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down_Atten(nn.Module):
    """CSA + Downscaling with stride2 conv then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.csa = CSA(in_channels, in_channels)
        #self.downconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        #self.conv = DoubleConv(in_channels, out_channels)
        self.atten_down = nn.Sequential(
            CSA(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.atten_down(x)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear='bilinear'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out

class Up_Atten(nn.Module):
    """CSA + Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.csa = CSA(in_channels // 2, in_channels // 2)
        """
        self.upsampling = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.PixelShuffle(upscale_factor=2)
        )

        self.upconv = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1),
        )
        """
        # self.up = nn.ConvTranspose2d(in_channels // 2 , in_channels // 2, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.conv = DoubleConv(in_channels, out_channels, mid_channels = in_channels // 2)
        #self.up = nn.ConvTranspose2d(in_channels //2 , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.csa(x1)
        #print(x1.size())
        #print(x2.size())
        #x1 = self.upsampling(x1)
        #x1 = self.upconv(x1)
        x1 = self.up(x1)
        #print(x1.size())
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetBaseModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear="bilinear"):
        super(UNetBaseModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels=3, out_channels=64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, input_):
        x1 = self.inc(input_)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4_2 = self.up1(x5, x4)
        x3_2 = self.up2(x4_2, x3)
        x2_2 = self.up3(x3_2, x2)
        x1_2 = self.up4(x2_2, x1)
        logits = self.outc(x1_2)
        out = torch.sigmoid(logits)

        return out, logits
        #return pred_masks

class UNetAttenModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNetAttenModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(in_channels=3, out_channels=64)
        self.down1 = Down_Atten(64, 128)
        self.down2 = Down_Atten(128, 256)
        self.down3 = Down_Atten(256, 512)
        #factor = 2 if bilinear else 1
        self.down4 = Down_Atten(512, 1024 // 2)
        self.up1 = Up_Atten(1024, 512 // 2)
        self.up2 = Up_Atten(512, 256 // 2)
        self.up3 = Up_Atten(256, 128 // 2)
        self.up4 = Up_Atten(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, input_):
        x1 = self.inc(input_)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4_2 = self.up1(x5, x4)
        x3_2 = self.up2(x4_2, x3)
        x2_2 = self.up3(x3_2, x2)
        x1_2 = self.up4(x2_2, x1)
        logits = self.outc(x1_2)
        out = torch.sigmoid(logits)

        return out, x1_2
        #return pred_masks

def test():
    dummy_input = torch.randn(2, 3, 96, 96)
    #model = UNetBaseModel()
    model = UNetAttenModel()
    out, logits = model(dummy_input)

    print(model)
    print('\nModel input shape :', dummy_input.size())
    print('Model output shape :', out.size())
    #print('ca_act_reg :', ca_act_reg)


if __name__ == '__main__':
    test()
