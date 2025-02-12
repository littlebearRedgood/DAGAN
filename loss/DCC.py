# -*- coding: utf-8 -*-
# @Author   : Honghao Xu
# @Time     : 2024/3/3 10:41
# @Function : Loss
import torch
import torch.nn as nn
import cv2
import numpy as np


class DoubleChannelColorLoss(nn.Module):
    def __init__(self):
        super(DoubleChannelColorLoss, self).__init__()

    def _rgb_to_hsv(self, rgb):
        x = rgb.float() / 255.0
        x = torch.clamp(x.permute(0, 2, 3, 1) * 255, 0, 255).byte()
        hsv = torch.from_numpy(np.array([cv2.cvtColor(i.cpu().numpy(), cv2.COLOR_RGB2HSV) for i in x])).permute(0, 3, 1, 2)
        return hsv.to(rgb.device)

    def forward(self, prediction, target):
        hsv_pred = self._rgb_to_hsv(prediction)
        hsv_target = self._rgb_to_hsv(target)
        dark_channel_pred = torch.min(hsv_pred[:, 2, :, :], dim=1, keepdim=True)[0]
        dark_channel_target = torch.min(hsv_target[:, 2, :, :], dim=1, keepdim=True)[0]
        dark_channel_loss = nn.MSELoss()(dark_channel_pred.float(), dark_channel_target.float())
        bright_channel_pred = torch.max(hsv_pred[:, 2, :, :], dim=1, keepdim=True)[0]
        bright_channel_target = torch.max(hsv_target[:, 2, :, :], dim=1, keepdim=True)[0]
        bright_channel_loss = nn.MSELoss()(bright_channel_pred.float(), bright_channel_target.float())
        combined_loss = dark_channel_loss + bright_channel_loss
        return combined_loss
