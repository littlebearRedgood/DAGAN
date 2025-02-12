import math
import random

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.utils import save_image


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


class VGG19_PercepLoss(nn.Module):
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'30': 'conv5_2'}  # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer] - pred_f[layer]) ** 2)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def pathGAN_():
    channels = 1
    input_size = 256
    num_downsampling = 5
    output_height = input_size // 2 ** num_downsampling
    output_width = input_size // 2 ** num_downsampling
    patch = (channels, output_height, output_width)
    return patch


def load_or_train(generator, discriminator, start_epoch):
    generator_path = f"./saveModels/generator/g_{start_epoch}.pth"
    discriminator_path = f"./saveModels/discriminator/d_{start_epoch}.pth"
    is_pretrained = False

    try:
        generator.load_state_dict(torch.load(generator_path))
        discriminator.load_state_dict(torch.load(discriminator_path))
        print(f"Successfully loaded pretrained models for epoch {start_epoch}")
        is_pretrained = True
    except FileNotFoundError:
        print(f"No pretrained model found for epoch {start_epoch}, training will start from scratch!")
        is_pretrained = False
        start_epoch = 0

    if is_pretrained:
        print("continue train--------------------------")
    return start_epoch


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        out = out
    elif mode == 1:
        out = np.flipud(out)
    elif mode == 2:
        out = np.rot90(out)
    elif mode == 3:
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        out = np.rot90(out, k=2)
    elif mode == 5:
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(out, k=3)
    elif mode == 7:
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def setup_optimizers_and_schedulers(generator, discriminator, lr, betas, step_size, gamma):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=step_size, gamma=gamma)
    scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=step_size, gamma=gamma)
    return optimizer_G, optimizer_D, scheduler_G, scheduler_D


def split(img):
    scale_factors = [0.125, 0.25, 0.5, 1]
    output = [F.interpolate(img, scale_factor=factor) for factor in scale_factors]
    return output


def save_sample_images(generator, x_test, Y_test, batch_index):
    generator.eval()
    test_index = random.randrange(1, 90)
    test_img = Variable(x_test[test_index, :, :, :]).unsqueeze(0).cuda()
    target_img = Variable(Y_test[test_index, :, :, :]).unsqueeze(0).cuda()
    generated_img = generator(test_img)
    imgx = generated_img[3].data
    imgy = target_img.data
    x = imgx[:, :, :, :]
    y = imgy[:, :, :, :]
    concat_img = torch.cat((x, y), -2)
    save_image(concat_img, f"images/results/{batch_index}.png", nrow=5, normalize=True)


def calculate_average(f, input1, input2):
    total = sum(f(input1[i], input2[i]) for i in range(4))
    return total / 4.0


def get_total_generator_loss(loss_GAN, loss_pixel, loss_mse, loss_ssim, loss_mac, loss_per, lambda_con, lambda_pixel,
                             lambda_mse, lambda_ssim, lambda_mac, lambda_per):
    return (lambda_con * loss_GAN + lambda_pixel * loss_pixel + lambda_mse * loss_mse + lambda_ssim * loss_ssim +
            lambda_mac * loss_mac + lambda_per * loss_per)


def get_total_generator_lossV1(loss_GAN, loss_pixel, loss_ssim, loss_con, loss_per, loss_dcc,
                               lambda_pixel, lambda_ssim, lambda_con, lambda_per, lambda_dcc):
    return (loss_GAN + lambda_pixel * loss_pixel + lambda_ssim * loss_ssim +
            lambda_con * loss_con + lambda_per * loss_per + lambda_dcc * loss_dcc)


def detach(fake):
    for i in range(4):
        fake[i] = fake[i].detach()
    return fake


def save_model_checkpoint(generator, discriminator, epoch):
    torch.save(generator.state_dict(), "saveModels/generator/generator_%d.pth" % epoch)
    torch.save(discriminator.state_dict(), "saveModels/discriminator/discriminator_%d.pth" % epoch)


def compute_mse(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    return mse


def compute_psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
