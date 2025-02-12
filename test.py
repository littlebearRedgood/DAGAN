import argparse
import json
import os
from thop import profile, clever_format
import time
import cv2
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from model.base_model.discriminatorV3 import Discriminator
from model.base_model.generatorV3 import Generator


def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def load_and_setup_model(params, generator_weight_path, discriminator_weight_path):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    generator = Generator(params).cuda()
    discriminator = Discriminator().cuda()
    generator.load_state_dict(torch.load(generator_weight_path))
    discriminator.load_state_dict(torch.load(discriminator_weight_path))
    generator.eval()
    discriminator.eval()
    input = torch.randn(1, 3, 256, 256).cuda()
    macs, params_num = profile(generator, inputs=(input,))
    macs, params_num = clever_format([macs, params_num], "%.3f")
    print(f'The generator has {params_num} parameters')
    print(f'The generator has {macs} FLOPs')

    return generator


def process_image(impath):
    imgx = cv2.imread(impath)
    if imgx is None:
        print("Failed to load image.")
    else:
        imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
        imgx = cv2.resize(imgx, (256, 256)).astype('float32')
        imgx = Variable(torch.from_numpy(imgx).permute(2, 0, 1).unsqueeze(0) / 255.0).cuda()
    return imgx


def generate_images(generator, path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path_list = sorted(os.listdir(path), key=lambda x: int(x.split('.')[0]))
    for item in path_list:
        img_path = os.path.join(path, item)
        print(f"Processing image at: {img_path}")
        img_input = process_image(img_path)

        # Calculate the inference time
        start_time = time.time()
        output = generator(img_input)[3].data
        inference_time = time.time() - start_time

        print(f"Inference time: {inference_time:.3f} seconds")

        save_image(output, os.path.join(output_dir, item), nrow=5, normalize=True)


def main():
    parser = argparse.ArgumentParser(description="Load weights and dataset paths from the command line.")
    parser.add_argument('--config', type=str, default='./config/config.json', help='Path to the config file')
    parser.add_argument('--generator_weights', type=str, required=True, help='Path to the generator weights file')
    parser.add_argument('--discriminator_weights', type=str, required=True, help='Path to the discriminator weights file')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the generated output')
    args = parser.parse_args()
    params = load_config(args.config)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.set_default_tensor_type(torch.FloatTensor)
    generator = load_and_setup_model(params, args.generator_weights, args.discriminator_weights)
    generate_images(generator, args.input_path, args.output_path)


if __name__ == '__main__':
    main()
