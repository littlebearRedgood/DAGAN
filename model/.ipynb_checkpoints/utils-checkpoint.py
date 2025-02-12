import math
import random

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
# from skimage.measure.simple_metrics import compare_psnr
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
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """

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
        # PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def pathGAN_():
    # 鉴别器的输出参数
    channels = 1  # 输出通道二进制真或假
    input_size = 256  # 输入图像大小
    num_downsampling = 5  # 下采样的次数

    # 计算鉴别器的输出大小
    output_height = input_size // 2 ** num_downsampling
    output_width = input_size // 2 ** num_downsampling
    patch = (channels, output_height, output_width)
    return patch


def load_or_train(generator, discriminator, start_epoch):
    generator_path = f"./saveModels/generator/g_{start_epoch}.pth"
    discriminator_path = f"./saveModels/discriminator/d_{start_epoch}.pth"
    is_pretrained = False

    try:
        # 尝试加载预训练模型权重
        generator.load_state_dict(torch.load(generator_path))
        discriminator.load_state_dict(torch.load(discriminator_path))
        print(f"Successfully loaded pretrained models for epoch {start_epoch}")
        is_pretrained = True
    except FileNotFoundError:
        print(f"No pretrained model found for epoch {start_epoch}, training will start from scratch!")
        is_pretrained = False
        start_epoch = 0

    if is_pretrained:
        print("继续训练--------------------------")
    # 执行训练过程...
    # 在此添加你的训练代码

    return start_epoch


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    # out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))
    # return out


def setup_optimizers_and_schedulers(generator, discriminator, lr, betas, step_size, gamma):
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    # Schedulers
    scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=step_size, gamma=gamma)
    scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=step_size, gamma=gamma)

    return optimizer_G, optimizer_D, scheduler_G, scheduler_D


def split(img):
    """
    对输入的图像进行不同比例的下采样，并将结果以及原图放入列表中返回。

    Args:
        img: 输入的原始图像。

    Returns:
        output: 包含原始图像以及不同比例下采样后的图像的列表。
    """
    scale_factors = [0.125, 0.25, 0.5, 1]
    output = [F.interpolate(img, scale_factor=factor) for factor in scale_factors]

    return output


def save_sample_images(generator, x_test, Y_test, batch_index):
    """
    这个函数使用生成器从验证集中生成一个样本并保存。

    参数：
        generator: 生成器模型
        x_test: 测试输入图像
        Y_test: 测试目标图像
        batch_index: 批次索引

    返回：
        无
    """
    generator.eval()  # 将生成器模型切换到评估模式

    # 从测试输入中随机获取一个索引
    test_index = random.randrange(1, 90)

    # 获取相应的测试图像张量(tensor)
    test_img = Variable(x_test[test_index, :, :, :]).unsqueeze(0).cuda()
    target_img = Variable(Y_test[test_index, :, :, :]).unsqueeze(0).cuda()

    # 使用生成器生成伪造图像
    generated_img = generator(test_img)

    # 将生成的图像和目标图像进行拼接
    imgx = generated_img[3].data
    imgy = target_img.data
    x = imgx[:, :, :, :]
    y = imgy[:, :, :, :]
    concat_img = torch.cat((x, y), -2)
    # 保存拼接后的图像
    save_image(concat_img, f"images/results/{batch_index}.png", nrow=5, normalize=True)


def calculate_average(f, input1, input2):
    total = sum(f(input1[i], input2[i]) for i in range(4))
    return total / 4.0


# def get_total_generator_loss(loss_GAN, loss_pixel, loss_ssim, loss_con, loss_lab, loss_lch,
#                              lambda_pixel, lambda_ssim, lambda_con, lambda_lab, lambda_lch):
#     return (loss_GAN + lambda_pixel * loss_pixel + lambda_ssim * loss_ssim +
#             lambda_con * loss_con + lambda_lab * loss_lab + lambda_lch * loss_lch)

# def get_total_generator_loss(loss_GAN, loss_pixel, loss_ssim, loss_con, loss_per, loss_dcc, lambda_pixel, lambda_ssim, lambda_con, lambda_per, lambda_dcc):
#     return (loss_GAN + lambda_pixel * loss_pixel + lambda_ssim * loss_ssim +
#             lambda_con * loss_con + lambda_per * loss_per + lambda_dcc * loss_dcc)

def get_total_generator_loss(loss_GAN, loss_pixel, loss_mse, loss_ssim, loss_mac, loss_per, lambda_con, lambda_pixel, lambda_mse, lambda_ssim, lambda_mac, lambda_per):
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


def save_model_checkpoint(generator, discriminator, epoch, name_):
    # Save model checkpoints
    # torch.save(generator.state_dict(), "saveModels/generator/generator_%d.pth" % epoch)
    # torch.save(discriminator.state_dict(), "saveModels/discriminator/discriminator_%d.pth" % epoch)
    torch.save(generator.state_dict(), f"saveModels/generator/generator_{name_}_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"saveModels/discriminator/discriminator_{name_}_{epoch}.pth")


def compute_mse(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    return mse


def compute_psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
