import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
import sys

class ConvLeakyBlock(nn.Module):
    """
    [conv - batch norm - leaky RELU] block
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, bn_momentum=0.9, leaky_slope=0.01):
        super(ConvLeakyBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=bn_momentum)
        self.activ = nn.LeakyReLU(negative_slope=leaky_slope)
    
    def forward(self, x):
        return self.activ(self.bn(self.conv(x)))

class ConvTransposeLeakyBlock(nn.Module):
    """
    [conv_transpose - batch norm - learky RELU] block
    """
    def __init__(self, in_channels, out_channels, output_size=None, kernel_size=3,
                 bias=True, bn_momentum=0.9, leaky_slope=0.01):
        super(ConvTransposeLeakyBlock, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=2, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=bn_momentum)
        self.activ = nn.LeakyReLU(negative_slope=leaky_slope)
            
    def forward(self, x, output_size=None):
        return self.activ(self.bn(self.conv_t(x, output_size=output_size)))
    
class LeakyUNET(nn.Module):
    def __init__(self):
        super(LeakyUNET, self).__init__()
        
        self.down_1 = nn.Sequential(
            ConvLeakyBlock(in_channels=1, out_channels=64),
            ConvLeakyBlock(in_channels=64, out_channels=64)
        )
        self.down_2 = nn.Sequential(
            ConvLeakyBlock(in_channels=64, out_channels=128),
            ConvLeakyBlock(in_channels=128, out_channels=128)
        )
        self.down_3 = nn.Sequential(
            ConvLeakyBlock(in_channels=128, out_channels=256),
            ConvLeakyBlock(in_channels=256, out_channels=256)
        )
        self.down_4 = nn.Sequential(
            ConvLeakyBlock(in_channels=256, out_channels=512),
            ConvLeakyBlock(in_channels=512, out_channels=512)
        )
        self.down_5 = nn.Sequential(
            ConvLeakyBlock(in_channels=512, out_channels=512),
            ConvLeakyBlock(in_channels=512, out_channels=512)
        )
        
        
        self.middle = nn.Sequential(
            ConvLeakyBlock(in_channels=512, out_channels=512),
            ConvLeakyBlock(in_channels=512, out_channels=512)
        )
        self.middle_t = ConvTransposeLeakyBlock(in_channels=512, out_channels=256)
        
        
        self.up_5 = nn.Sequential(
            ConvLeakyBlock(in_channels=768, out_channels=512),
            ConvLeakyBlock(in_channels=512, out_channels=512)
        )
        self.up_5_t = ConvTransposeLeakyBlock(in_channels=512, out_channels=256)
        self.up_4 = nn.Sequential(
            ConvLeakyBlock(in_channels=768, out_channels=512),
            ConvLeakyBlock(in_channels=512, out_channels=512)
        )
        self.up_4_t = ConvTransposeLeakyBlock(in_channels=512, out_channels=128)
        self.up_3 = nn.Sequential(
            ConvLeakyBlock(in_channels=384, out_channels=256),
            ConvLeakyBlock(in_channels=256, out_channels=256)
        )
        self.up_3_t = ConvTransposeLeakyBlock(in_channels=256, out_channels=64)
        self.up_2 = nn.Sequential(
            ConvLeakyBlock(in_channels=192, out_channels=128),
            ConvLeakyBlock(in_channels=128, out_channels=128)
        )
        self.up_2_t = ConvTransposeLeakyBlock(in_channels=128, out_channels=32)
        self.up_1 = nn.Sequential(
            ConvLeakyBlock(in_channels=96, out_channels=64),
            ConvLeakyBlock(in_channels=64, out_channels=1)
        )
    
    def forward(self, x):
        down1 = self.down_1(x) # (1 x 256 x 256 -> 64 x 256 x 256)
        out = F.max_pool2d(down1, kernel_size=2, stride=2) # (64 x 256 x 256 -> 64 x 128 x 128)
        
        down2 = self.down_2(out) # (64 x 128 x 128 -> 128 x 128 x 128)
        out = F.max_pool2d(down2, kernel_size=2, stride=2) # (128 x 128 x 128 -> 128 x 64 x 64)
        
        down3 = self.down_3(out) # (128 x 64 x 64 -> 256 x 64 x 64)
        out = F.max_pool2d(down3, kernel_size=2, stride=2) # (256 x 64 x 64 -> 256 x 32 x 32)
        
        down4 = self.down_4(out) # (256 x 32 x 32 -> 512 x 32 x 32)
        out = F.max_pool2d(down4, kernel_size=2, stride=2) # (512 x 32 x 32 -> 512 x 16 x 16)
        
        down5 = self.down_5(out) # (512 x 16 x 16 -> 512 x 16 x 16)
        out = F.max_pool2d(down5, kernel_size=2, stride=2) # (512 x 16 x 16 -> 512 x 8 x 8)

        out = self.middle(out) # (512 x 8 x 8 -> 512 x 8 x 8)
        out = self.middle_t(out, output_size=down5.size()) # (512 x 8 x 8 -> 256 x 16 x 16)
        
        out = torch.cat([down5, out], 1) # (512 x 16 x 16 concat 256 x 16 x 16 -> 768 x 16 x 16)
        out = self.up_5(out) # (768 x 16 x 16 -> 512 x 16 x 16)
        out = self.up_5_t(out, output_size=down4.size()) # (512 x 16 x 16 -> 256 x 32 x 32)
        
        out = torch.cat([down4, out], 1) # (512 x 32 x 32 concat 256 x 32 x 32 -> 768 x 32 x 32)
        out = self.up_4(out) # (768 x 32 x 32 -> 512 x 32 x 32)
        out = self.up_4_t(out, output_size=down3.size()) # (512 x 32 x 32 -> 128 x 64 x 64)
        
        out = torch.cat([down3, out], 1) # (256 x 64 x 64 concat 128 x 64 x 64 -> 384 x 64 x 64)
        out = self.up_3(out) # (384 x 64 x 64 -> 256 x 64 x 64)
        out = self.up_3_t(out, output_size=down2.size()) # (256 x 64 x 64 -> 64 x 128 x 128)
        
        out = torch.cat([down2, out], 1) # (128 x 128 x 128 concat 64 x 128 x 128 -> 192 x 128 x 128)
        out = self.up_2(out) # (192 x 128 x 128 -> 128 x 128 x 128)
        out = self.up_2_t(out, output_size=down1.size()) # (128 x 128 x 128 -> 32 X 256 x 256)
        
        out = torch.cat([down1, out], 1) # (64 x 256 x 256 concat 32 x 256 x 256 -> 96 x 256 x 256)
        out = self.up_1(out) # (96 x 256 x 256 -> 1 x 256 x 256)
        
        return out

class LeakyUNETWithResnet34(nn.Module):
    def __init__(self):
        super(LeakyUNETWithResnet34, self).__init__()

        self.encoder = torchvision.models.resnet34(pretrained=True)
        self.num_filters = 32

        self.down_1 = nn.Sequential(
          self.encoder.conv1.apply(self.squeeze_weights),
          self.encoder.bn1,
	  nn.LeakyReLU(negative_slope=0.01),
          nn.MaxPool2d(2, 2)
	) 
        self.down_2 =self.encoder.layer1
        self.down_3 = self.encoder.layer2
        self.down_4 = self.encoder.layer3
        self.down_5 = self.encoder.layer4
        
        self.middle = nn.Sequential(
            ConvLeakyBlock(in_channels=512, out_channels=512),
            ConvLeakyBlock(in_channels=512, out_channels=512)
        )
        self.middle_t = ConvTransposeLeakyBlock(in_channels=512, out_channels=256, kernel_size=4)
        
        
        self.up_5 = nn.Sequential(
            ConvLeakyBlock(in_channels=768, out_channels=512),
            ConvLeakyBlock(in_channels=512, out_channels=512)
        )
        self.up_5_t = ConvTransposeLeakyBlock(in_channels=512, out_channels=512, kernel_size=4)
        self.up_4 = nn.Sequential(
            ConvLeakyBlock(in_channels=768, out_channels=512),
            ConvLeakyBlock(in_channels=512, out_channels=512)
        )
        self.up_4_t = ConvTransposeLeakyBlock(in_channels=512, out_channels=256, kernel_size=4)
        self.up_3 = nn.Sequential(
            ConvLeakyBlock(in_channels=384, out_channels=256),
            ConvLeakyBlock(in_channels=256, out_channels=256)
        )
        self.up_3_t = ConvTransposeLeakyBlock(in_channels=256, out_channels=128, kernel_size=4)
        self.up_2 = nn.Sequential(
            ConvLeakyBlock(in_channels=192, out_channels=128),
            ConvLeakyBlock(in_channels=128, out_channels=128)
        )
        self.up_2_t = ConvTransposeLeakyBlock(in_channels=128, out_channels=96, kernel_size=4)
        self.up_1 = nn.Sequential(
            ConvLeakyBlock(in_channels=96, out_channels=64),
            ConvLeakyBlock(in_channels=64, out_channels=64)
        )
        self.up_1_t = ConvTransposeLeakyBlock(in_channels=64, out_channels=1, kernel_size=4)

    def squeeze_weights(self, m):
        m.weight.data = m.weight.data.sum(dim=1)[:,None]
        m.in_channels = 1
    
    def forward(self, x):
        down1 = self.down_1(x) # (1 x 256 x 256 -> 64 x 256 x 256)
        
        down2 = self.down_2(down1) # (64 x 128 x 128 -> 128 x 128 x 128)
        # out = F.max_pool2d(down2, kernel_size=2, stride=2) # (128 x 128 x 128 -> 128 x 64 x 64)
        
        down3 = self.down_3(down2) # (128 x 64 x 64 -> 256 x 64 x 64)
        # out = F.max_pool2d(down3, kernel_size=2, stride=2) # (256 x 64 x 64 -> 256 x 32 x 32)
        
        down4 = self.down_4(down3) # (256 x 32 x 32 -> 512 x 32 x 32)
        # out = F.max_pool2d(down4, kernel_size=2, stride=2) # (512 x 32 x 32 -> 512 x 16 x 16)
        
        down5 = self.down_5(down4) # (512 x 16 x 16 -> 512 x 16 x 16)
        out = F.max_pool2d(down5, kernel_size=2, stride=2) # (512 x 16 x 16 -> 512 x 8 x 8)

        out = self.middle(out) # (512 x 8 x 8 -> 512 x 8 x 8)
        out = self.middle_t(out)
 
        out = torch.cat([down5, out], 1) # (512 x 16 x 16 concat 256 x 16 x 16 -> 768 x 16 x 16)
        out = self.up_5(out) # (768 x 16 x 16 -> 512 x 16 x 16)
        out = self.up_5_t(out)

        out = torch.cat([down4, out], 1) # (512 x 32 x 32 concat 256 x 32 x 32 -> 768 x 32 x 32)
        out = self.up_4(out) # (768 x 32 x 32 -> 512 x 32 x 32)
        out = self.up_4_t(out)
        
        out = torch.cat([down3, out], 1) # (256 x 64 x 64 concat 128 x 64 x 64 -> 384 x 64 x 64)
        out = self.up_3(out) # (384 x 64 x 64 -> 256 x 64 x 64)
        out = self.up_3_t(out)
        
        out = torch.cat([down2, out], 1) # (128 x 128 x 128 concat 64 x 128 x 128 -> 192 x 128 x 128)
        out = self.up_2(out) # (192 x 128 x 128 -> 128 x 128 x 128)
        out = self.up_2_t(out)

        # out = torch.cat([down1, out], 1) # (64 x 256 x 256 concat 32 x 256 x 256 -> 96 x 256 x 256)
        out = self.up_1(out) # (96 x 256 x 256 -> 1 x 256 x 256)
        out = self.up_1_t(out)
        
        return out
