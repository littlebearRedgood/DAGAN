import json
import os
import time
import cv2
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from model.base_model.discriminator import Discriminator
from model.base_model.generator import Generator


def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def load_and_setup_model(params):
    # cuda = True if torch.cuda.is_available() else False
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    generator = Generator(params).cuda()
    discriminator = Discriminator().cuda()
    generator.load_state_dict(torch.load("./saveModels/generator_epoch.pth"))
    discriminator.load_state_dict(torch.load("./saveModels/discriminator_epoch.pth"))
    generator.eval()
    discriminator.eval()
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
        start_time = time.time()
        output = generator(img_input)[3].data
        inference_time = time.time() - start_time
        # print(f"Inference time: {inference_time:.3f} seconds")
        save_image(output, os.path.join(output_dir, item), nrow=5, normalize=True)


def main():
    params = load_config('./config/config.json')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.set_default_tensor_type(torch.FloatTensor)
    generator = load_and_setup_model(params)
    generate_images(generator, './DataSets/UIEB/raw/', './DataSets/UIEB/our/')


if __name__ == '__main__':
    main()
