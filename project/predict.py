"""Model predict."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 09月 09日 星期三 23:56:45 CST
# ***
# ************************************************************************************/
#
import os
import glob
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.utils as utils
from PIL import Image
from model import get_model, model_load, model_setenv
from tqdm import tqdm
import pdb

if __name__ == "__main__":
    """Predict."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="output/ImageClean.pth", help="checkpint file")
    parser.add_argument('--input', type=str, default="dataset/Polyu/CroppedImages/Sony_*real.JPG", help="input image")
    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)
    model.eval()

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model = amp.initialize(model, opt_level="O1")

    totensor = transforms.ToTensor()
    # toimage = transforms.ToPILImage()

    image_filenames = glob.glob(args.input)
    progress_bar = tqdm(total = len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        # GT image
        head, tail = os.path.split(filename)
        image_path = os.path.join(head, tail.replace('real', 'mean'))
        clean_image = Image.open(image_path).convert("RGB")
        gt_tensor = totensor(clean_image).unsqueeze(0).to(device)

        image = Image.open(filename).convert("RGB")
        input_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            noise_level_est, output_tensor = model(input_tensor)

        output_tensor.clamp_(0, 1.0)

        grid = utils.make_grid(torch.cat([gt_tensor, input_tensor, output_tensor], dim=0), nrow=3)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        image = Image.fromarray(ndarr)
        image.show()
