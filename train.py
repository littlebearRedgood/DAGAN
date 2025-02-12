# -*- coding: utf-8 -*-
# @Author   : Honghao Xu
# @Time     : 2024/3/1 14:59
# @Function : train
import argparse
import csv
import datetime
import logging
import os
import sys
import time
import json
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from data.dataset_handler import DatasetHandler
from loss.init_losses import init_loss_funcs
from model.base_model.generator import Generator
from model.base_model.discriminator import Discriminator
from model.utils import pathGAN_, load_or_train, setup_optimizers_and_schedulers, split, batch_PSNR, save_sample_images, \
    calculate_average, get_total_generator_loss, detach, save_model_checkpoint


def main():
    global real_B, fake_B
    with open('./config/config.json', 'r') as f:
        params = json.load(f)
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.set_default_tensor_type(torch.FloatTensor)
    # print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', help='the number of batch', type=int, default=16)
    parser.add_argument('-e', '--epoch', help='the number of training', type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument('--sample_interval', type=int, default=100)
    parser.add_argument('-r', '--resume', help='the choice of resume', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.5, 0.999], help='Betas for Adam optimizer')
    parser.add_argument('--step_size', type=int, default=40, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.8, help='Gamma for learning rate scheduler')

    args = parser.parse_args()

    logging.basicConfig(filename='application.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    prev_time = time.time()
    n_epochs = args.epoch
    sample_interval = args.sample_interval
    checkpoint_interval = args.checkpoint_interval
    handler = DatasetHandler(train_path='./DataSets/data/train/', test_path='./DataSets/data/test/')
    x_train, y_train, x_test, y_test = handler.init_dataset()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch, shuffle=True, num_workers=4)
    criterion_GAN, criterion_pixelwise, mse_loss, ssim_loss, l_mac, l_dcc, l_vgg, l_lab, l_lch, l_per, l_pers, lambda_con, lambda_pixel, lambda_mse, lambda_ssim, lambda_mac, lambda_dcc, lambda_vgg, lambda_lab, lambda_lch, lambda_per, lambda_pers = init_loss_funcs()
    patch = pathGAN_()
    generator = Generator(params).cuda()
    discriminator = Discriminator().cuda()
    load_or_train(generator, discriminator, start_epoch=n_epochs)
    optimizer_G, optimizer_D, scheduler_G, scheduler_D = setup_optimizers_and_schedulers(generator, discriminator,
                                                                                         args.lr, args.betas,
                                                                                         args.step_size, args.gamma)
    for epoch in range(n_epochs+1):
        for i, batch in enumerate(loader):
            real_A = Variable(batch[0]).cuda()
            real_B = Variable(batch[1]).cuda()
            real_A1 = split(real_A)
            real_B1 = split(real_B)
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False).cuda()
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False).cuda()
            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A1).cuda()
            loss_GAN = criterion_GAN(pred_fake, valid).cuda()
            loss_pixel = calculate_average(criterion_pixelwise, fake_B, real_B1)
            loss_mse = calculate_average(mse_loss, fake_B, real_B1)
            loss_ssim = -calculate_average(ssim_loss, fake_B, real_B1)
            loss_mac = calculate_average(l_mac, fake_B, real_B1)
            loss_vgg = calculate_average(l_vgg, fake_B, real_B1)
            loss_per = calculate_average(l_per, fake_B, real_B1)
            ssim_value = - loss_ssim.item()
            loss_G = get_total_generator_loss(loss_GAN, loss_pixel, loss_mse, loss_ssim, loss_mac, loss_per,
                                              lambda_con, lambda_pixel, lambda_mse, lambda_ssim,
                                              lambda_mac, lambda_per)
            loss_G.backward(retain_graph=True)
            optimizer_G.step()
            optimizer_D.zero_grad()
            pred_real = discriminator(real_B1, real_A1).cuda()
            loss_real = criterion_GAN(pred_real, valid).cuda()
            fake_B = detach(fake_B)
            pred_fake1 = discriminator(fake_B, real_A1).cuda()
            loss_fake = criterion_GAN(pred_fake1, fake).cuda()
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward(retain_graph=True)
            optimizer_D.step()
        batches_done = epoch * len(loader) + i
        batches_left = n_epochs * len(loader) - batches_done
        out_train = torch.clamp(fake_B[3], 0., 1.)
        psnr_train = batch_PSNR(out_train, real_B, 1.)
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        logging.info(
           f'[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(loader)}][PSNR: {psnr_train}] [SSIM: {ssim_value}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}],[mac: {0.0001 * loss_mac.item()}], [pixel: {0.0001 * loss_pixel.item()}],[VGG_loss: {0.01 * loss_vgg.item()}], [adv: {loss_GAN.item()}] ETA: {time_left}')
        if batches_done % sample_interval == 0:
            save_sample_images(generator, x_test, y_test, batches_done)
        scheduler_G.step()
        scheduler_D.step()
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            save_model_checkpoint(generator, discriminator, epoch)


if __name__ == '__main__':
    main()
