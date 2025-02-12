import torch
from torch import nn
from torch.nn import ModuleList, LeakyReLU
from model.block import from_rgb, _equalized_conv2d, PixelwiseNorm, DisGeneralConvBlock, DisFinalBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, use_eql=True):
        super(Discriminator, self).__init__()
        self.use_eql = use_eql
        self.in_channels = in_channels
        self.rgb_to_feature1 = ModuleList([from_rgb(32), from_rgb(64), from_rgb(128)])
        self.rgb_to_feature2 = ModuleList([from_rgb(32), from_rgb(64), from_rgb(128)])
        self.layer = _equalized_conv2d(self.in_channels * 2, 64, (1, 1), bias=True)
        self.pixNorm = PixelwiseNorm()
        self.lrelu = LeakyReLU(0.2)
        self.layer0 = DisGeneralConvBlock(64, 64, use_eql=self.use_eql)
        self.layer1 = DisGeneralConvBlock(128, 128, use_eql=self.use_eql)
        self.layer2 = DisGeneralConvBlock(256, 256, use_eql=self.use_eql)
        self.layer3 = DisGeneralConvBlock(512, 512, use_eql=self.use_eql)
        self.layer4 = DisFinalBlock(512, use_eql=self.use_eql)

    def forward(self, img_A, inputs):
        x = torch.cat((img_A[3], inputs[3]), 1)
        y = self.pixNorm(self.lrelu(self.layer(x)))
        y = self.layer0(y)
        x1 = self.rgb_to_feature1[0](img_A[2])
        x2 = self.rgb_to_feature2[0](inputs[2])
        x = torch.cat((x1, x2), 1)
        y = torch.cat((x, y), 1)
        y = self.layer1(y)
        x1 = self.rgb_to_feature1[1](img_A[1])
        x2 = self.rgb_to_feature2[1](inputs[1])
        x = torch.cat((x1, x2), 1)
        y = torch.cat((x, y), 1)
        y = self.layer2(y)
        x1 = self.rgb_to_feature1[2](img_A[0])
        x2 = self.rgb_to_feature2[2](inputs[0])
        x = torch.cat((x1, x2), 1)
        y = torch.cat((x, y), 1)
        y = self.layer3(y)
        y = self.layer4(y)
        return y
