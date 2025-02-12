# -*- coding: utf-8 -*-
"""
Author: Honghao Xu
Date: 2024/3/1 15:49
Description: Initialize loss functions for the training process.
"""
import torch
import torch.nn as nn

from loss.DCC import DoubleChannelColorLoss
from loss.MultiAspectColorLoss import MultiAspectColor
from loss.lab import lab_Loss
from loss.lch import lch_Loss
from loss.perceptual import PerceptualLoss, PerceptualLoss2
from model.utils import VGG19_PercepLoss
from measure import pytorch_ssim


def init_loss_funcs():
    criterion_gan = nn.MSELoss(reduction='sum')
    criterion_pixelwise = nn.L1Loss(reduction='sum')
    mse_loss = nn.MSELoss(reduction='sum')
    ssim_loss = pytorch_ssim.SSIM()
    l_mac = MultiAspectColor(0.1, 0.1)
    l_dcc = DoubleChannelColorLoss()
    l_vgg = VGG19_PercepLoss()
    l_lab = lab_Loss()
    l_lch = lch_Loss()
    l_per = PerceptualLoss()
    l_pers = PerceptualLoss2()
    if torch.cuda.is_available():
        criterion_gan = criterion_gan.cuda()
        criterion_pixelwise = criterion_pixelwise.cuda()
        mse_loss = mse_loss.cuda()
        ssim_loss = ssim_loss.cuda()
        l_mac = l_mac.cuda()
        l_dcc = l_dcc.cuda()
        l_vgg = l_vgg.cuda()
        l_lab = l_lab.cuda()
        l_lch = l_lch.cuda()
        l_per = l_per.cuda()
        l_pers = l_pers.cuda()
    lambda_con = 0.1
    lambda_pixel = 0.1
    lambda_mse = 0.1
    lambda_ssim = 0.1
    lambda_mac = 0.1
    lambda_dcc = 0.1
    lambda_vgg = 0.1
    lambda_lab = 0.1
    lambda_lch = 0.1
    lambda_per = 0.1
    lambda_pers = 0.1
    return criterion_gan, criterion_pixelwise, mse_loss, ssim_loss, l_mac, l_dcc, l_vgg, l_lab, l_lch, l_per, l_pers, \
           lambda_con, lambda_pixel, lambda_mse, lambda_ssim, lambda_mac, lambda_dcc, lambda_vgg, lambda_lab, lambda_lch, lambda_per, lambda_pers