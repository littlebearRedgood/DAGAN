# -*- coding: utf-8 -*-
# @Author   : Honghao Xu
# @Time     : 2024/3/6 15:42
# @Function : Loss
import torch.nn as nn
from loss.DCC import DoubleChannelColorLoss
from loss.lab import lab_Loss
from loss.lch import lch_Loss


class MultiAspectColor(nn.Module):
    def __init__(self, lab_weigth, lch_weigth):
        super(MultiAspectColor, self).__init__()
        self.double_channel_loss = DoubleChannelColorLoss()
        self.lab_loss = lab_Loss(lab_weigth)
        self.lch_loss = lch_Loss(lch_weigth)

    def forward(self, prediction, target):
        double_channel_loss_value = self.double_channel_loss(prediction, target)
        lab_loss_value = self.lab_loss(prediction, target)
        lch_loss_value = self.lch_loss(prediction, target)
        total_loss = double_channel_loss_value + lab_loss_value + lch_loss_value
        return total_loss