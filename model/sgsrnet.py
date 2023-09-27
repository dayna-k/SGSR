import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        self.b1 = nn.Sequential(nn.Conv2d(1*filters, filters, 3, 1, 1, bias=True), nn.LeakyReLU())
        self.b2 = nn.Sequential(nn.Conv2d(2*filters, filters, 3, 1, 1, bias=True), nn.LeakyReLU())
        self.b3 = nn.Sequential(nn.Conv2d(3*filters, filters, 3, 1, 1, bias=True), nn.LeakyReLU())
        self.b4 = nn.Sequential(nn.Conv2d(4*filters, filters, 3, 1, 1, bias=True), nn.LeakyReLU())
        self.b5 = nn.Sequential(nn.Conv2d(5*filters, filters, 3, 1, 1, bias=True))
        self.blocks = nn.ModuleList([self.b1, self.b2, self.b3, self.b4, self.b5])
        

    def forward(self, x):
        inputs = x
        # input in 1, block weight in 0
        for block in self.blocks:
            # print("inputs: ", inputs.get_device())
            if x.get_device() != block[0].weight.get_device():
                print("###### Incorrect ####### x: ", x.get_device(), "/ block: ", block[0].weight.get_device())
            
            out = block(inputs) #####################
            inputs = torch.cat([inputs, out], 1)
            
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x #####################



class Guided_Attn(nn.Module):
    """ Guided attention Layer"""

    def __init__(self, in_dim, activation):
        super(Guided_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.alpha1 = nn.Parameter(torch.rand(1))
        self.alpha2 = nn.Parameter(torch.rand(1))
        self.softmax = nn.Softmax(dim=-1)  #
        
        self.mask_key_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=in_dim // 8, kernel_size=1)
        )

        self.mask_query_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=in_dim // 8, kernel_size=1)
        )

        self.convadd = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        )
        self.BilinearDown = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.BilinearUp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, x, mask):
        """
            inputs :
                x : input feature maps( B X C X W X H)
                mask : input mask (B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        mask_query = self.mask_query_conv(mask) # torch.Size([1, 8, 96, 96])
        mask_key = self.mask_key_conv(mask) # torch.Size([1, 8, 96, 96])

        x_query = self.query_conv(x)
        x_key = self.key_conv(x)

        proj_query = torch.mul(mask_query, x_query) * self.alpha1 + x_query * (1-self.alpha1)
        proj_query = self.BilinearDown(proj_query)
        proj_query = proj_query.view(m_batchsize, -1, int(width/2) * int(height/2)).permute(0, 2, 1)

        proj_key = torch.mul(mask_key, x_key) * self.alpha2 + x_key* (1-self.alpha2)
        proj_key = self.BilinearDown(proj_key)
        proj_key = proj_key.view(m_batchsize, -1, int(width/2) * int(height/2))

        
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x)
        proj_value = self.BilinearDown(proj_value)
        proj_value = proj_value.view(m_batchsize, -1, int(width/2) * int(height/2))  # B X C X N torch.Size([1, 64, 9216])

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # torch.Size([1, 64, 9216])
        out = out.view(m_batchsize, C, int(width/2), int(height/2)) # torch.Size([1, 64, 96, 96])
        # Upsample
        out = self.BilinearUp(out)
        out = self.convadd(out)
        out = out + x
        # diff = out - x
        #print("gamma: ", self.gamma)
        #return out, weighted_attention, attention, attention_mask, diff
        return out



class GeneratorRRDB_SGSR(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB_SGSR, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

        self.attn1 = Guided_Attn(filters, 'relu')

    def forward(self, x, feat):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)       #####################
        out2 = self.conv2(out)
        img_feat = torch.add(out1, out2)

        # feat 128 channel -> 64channel
        # out_feat, w_atten, atten, atten_mask, diff = self.attn1(img_feat, feat)
        # out_feat, w_atten, atten, atten_mask, diff = self.attn1(img_feat, feat)
        out_feat = self.attn1(img_feat, feat)

        out = self.upsampling(out_feat)
        out = self.conv3(out)

        # return out, w_atten, atten, atten_mask, img_feat, out_feat, diff
        # return out, w_atten, atten, atten_mask
        return out



class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (3, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.ModuleList(layers)

        layers = nn.ModuleList([])
        in_filters = 3  # target(3) + importance map

        for i, out_filters in enumerate([64, 128, 128, 128]):
            ## 64, 128, 256, 512 ---> D>>G
            ## 64, 128, 256, 256 ---> D>>G
            ## 64, 128, 128, 128 ---> D>>G

            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 3, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

