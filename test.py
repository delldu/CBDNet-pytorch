from __future__ import division
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import re
import cv2

import pdb

from utils import *
from model import *
from PIL import Image

from apex import amp
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Image Clean')
parser.add_argument('--cpu', nargs='?', const=1, help = 'Use CPU')
args = parser.parse_args()

input_dir = 'test/input/'
result_dir = 'test/output/'
checkpoint_filename = 'checkpoint/CBDNet.pth'


image_filenames = glob.glob(input_dir + '*.png')

# model load
model = CBDNet()

if os.path.exists(checkpoint_filename):
    # load existing model
    model_info = torch.load(checkpoint_filename)
    print('==> loading existing model:', checkpoint_filename)

    model.load_state_dict(model_info)
    #['state_dict'])
else:
    print('Error: No trained model detected!')
    exit(1)

if not args.cpu:
    print('Using GPU!')
    model.cuda()
else:
    print('Using CPU!')
model.eval()

model = amp.initialize(model, opt_level= "O1")

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

# pdb.set_trace()
progress_bar = tqdm(total = len(image_filenames))

for index, filename in enumerate(image_filenames):
    progress_bar.update(1)

    with torch.no_grad():
        noisy_img = cv2.imread(filename)
        noisy_img = noisy_img[:, :, ::-1] / 255.0
        noisy_img = np.array(noisy_img).astype('float32')

        temp_noisy_img = noisy_img
        temp_noisy_img_chw = hwc_to_chw(temp_noisy_img)

        input_var = torch.from_numpy(temp_noisy_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
        if not args.cpu:
            input_var = input_var.cuda()

        output = model(input_var)

        output_np = output.squeeze().cpu().detach().numpy()

        output_np = chw_to_hwc(np.clip(output_np, 0, 1))
        temp = np.concatenate((temp_noisy_img, output_np), axis=1)

        temp = temp * 255
        img = Image.fromarray(temp.astype('uint8')).convert('RGB')
        img.save(result_dir + os.path.basename(filename))
