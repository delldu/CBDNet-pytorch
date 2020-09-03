import os
import argparse
import glob
import pdb
import torch
from torchvision import transforms
from model import *
from PIL import Image
from apex import amp
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Image Clean')
parser.add_argument('--cpu', nargs='?', const=1, help = 'Use CPU')
args = parser.parse_args()

input_dir = 'dataset/test/'
result_dir = 'test/output/'
checkpoint_filename = 'checkpoint/CBDNet.pth'
# checkpoint_filename = 'checkpoint/CBDNet_JPEG.pth'

# model load
model = CBDNet()
if os.path.exists(checkpoint_filename):
    # load existing model
    model_info = torch.load(checkpoint_filename)
    print('==> loading existing model:', checkpoint_filename)
    model.load_state_dict(model_info)
else:
    print('Error: No trained model detected!')
    exit(1)

if not args.cpu:
    model.cuda()
    model = amp.initialize(model, opt_level= "O1")
model.eval()

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

totensor = transforms.ToTensor()
toimage = transforms.ToPILImage()

image_filenames = glob.glob(input_dir + '*.bmp')
progress_bar = tqdm(total = len(image_filenames))

for index, filename in enumerate(image_filenames):
    progress_bar.update(1)

    noisy_img = Image.open(filename).convert("RGB")
    w, h = noisy_img.size
    w = (w // 16) * 8
    h = (h // 16) * 8
    noisy_img = noisy_img.resize((w, h))

    input_tensor = totensor(noisy_img).unsqueeze(0)

    if not args.cpu:
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        _, output_tensor = model(input_tensor)

    del input_tensor
    output_tensor = output_tensor.clamp(0, 1.0).cpu()

    # temp = torch.cat((input_tensor.squeeze(), output_tensor.squeeze()), 2).cpu()
    # toimage(temp).save(result_dir + os.path.basename(filename))

    toimage(output_tensor.squeeze()).save(result_dir + os.path.basename(filename))
