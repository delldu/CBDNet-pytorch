"""Create model."""
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
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import pdb

# The following core come from https://github.com/b03901165Shih/CBDNet-pytorch-inference.git
# Thank the authors, i love you !!!

class ImageCleanModel(nn.Module):
    """ImageClean Model."""

    def __init__(self):
        """Init model."""
        super(ImageCleanModel, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.E01 = nn.Conv2d(3, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.E02 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.E03 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.E04 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.E05 = nn.Conv2d(32, 3, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))

        # input
        self.DS01_layer00 = nn.Conv2d(6, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS01_layer01 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS01_layer02 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS01_layer03 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))

        self.DS02 = nn.Conv2d(64, 256, kernel_size=[2, 2], stride=(2, 2))
        self.DS02_layer00_cf = nn.Conv2d(256, 128, kernel_size=[1, 1], stride=(1, 1))
        self.DS02_layer00 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS02_layer01 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS02_layer02 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))

        self.DS03 = nn.Conv2d(128, 512, kernel_size=[2, 2], stride=(2, 2))
        self.DS03_layer00_cf = nn.Conv2d(512, 256, kernel_size=[1, 1], stride=(1, 1))
        self.DS03_layer00 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS03_layer01 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS03_layer02 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        
        self.UPS03_layer00 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.UPS03_layer01 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.UPS03_layer02 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.UPS03_layer03 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))

        self.USP02 = nn.ConvTranspose2d(512, 128, kernel_size=[2, 2], stride=(2, 2), bias=False)
        self.US02_layer00 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.US02_layer01 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.US02_layer02 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))

        self.USP01 = nn.ConvTranspose2d(256, 64, kernel_size=[2, 2], stride=(2, 2), bias=False)
        self.US01_layer00 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.US01_layer01 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))

        # output
        self.US01_layer02 = nn.Conv2d(64, 3, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))

    def forward(self, input):
        x = self.E01(input)
        self.relu(x)
        x = self.E02(x)
        self.relu(x)
        x = self.E03(x)
        self.relu(x)
        x = self.E04(x)
        self.relu(x)
        x = self.E05(x)
        self.relu(x)
        noise_level = x
        x = torch.cat((input, noise_level), dim=1)

        x = self.DS01_layer00(x)
        self.relu(x)
        x = self.DS01_layer01(x)
        self.relu(x)
        x = self.DS01_layer02(x)
        self.relu(x)
        x = self.DS01_layer03(x)
        self.relu(x)
        down1_result = x

        x = self.DS02(down1_result)
        x = self.DS02_layer00_cf(x)
        x = self.DS02_layer00(x)
        self.relu(x)
        x = self.DS02_layer01(x)
        self.relu(x)
        x = self.DS02_layer02(x)
        self.relu(x)

        down2_result = x
        x = self.DS03(down2_result)
        x = self.DS03_layer00_cf(x)
        x = self.DS03_layer00(x)
        self.relu(x)
        x = self.DS03_layer01(x)
        self.relu(x)
        x = self.DS03_layer02(x)
        self.relu(x)
        x = self.UPS03_layer00(x)
        self.relu(x)
        x = self.UPS03_layer01(x)
        self.relu(x)
        x = self.UPS03_layer02(x)
        self.relu(x)
        x = self.UPS03_layer03(x)
        self.relu(x)

        x = self.USP02(x)

        x = torch.add(x, 1, down2_result)
        del down2_result

        x = self.US02_layer00(x)
        self.relu(x)
        x = self.US02_layer01(x)
        self.relu(x)
        x = self.US02_layer02(x)
        self.relu(x)
        x = self.USP01(x)
        x = torch.add(x, 1, down1_result)
        del down1_result

        x = self.US01_layer00(x)
        self.relu(x)
        x = self.US01_layer01(x)
        self.relu(x)
        x = self.US01_layer02(x)
        y = torch.add(input, 1, x)
        
        del x

        return noise_level, y


class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        h_tv = torch.pow(
            (est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow(
            (est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        # n = gt_noise - est_noise
        # a = torch.abs(0.3 - F.relu(n))
        # b = torch.pow(n, 2)
        # torch.mul(a, b)
        loss = torch.mean(torch.pow((out_image - gt_image), 2)) + \
                if_asym * 0.5 * torch.mean(torch.mul(torch.abs(0.3 - F.relu(gt_noise - est_noise)), torch.pow(est_noise - gt_noise, 2))) + \
                0.05 * tvloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def model_load(model, path):
    """Load model."""
    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""
    torch.save(model.state_dict(), path)

def model_export():
    """Export model to onnx."""

    import onnx
    from onnx import optimizer

    # xxxx--modify here
    onnx_file = "model.onnx"
    weight_file = "checkpoint/weight.pth"

    # 1. Load model
    print("Loading model ...")
    model = ImageCleanModel()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    # xxxx--modify here
    dummy_input = torch.randn(1, 3, 512, 512)
    input_names = [ "input" ]
    output_names = [ "output" ]
    torch.onnx.export(model, dummy_input, onnx_file,
                    input_names=input_names, 
                    output_names=output_names,
                    verbose=True,
                    opset_version=11,
                    keep_initializers_as_inputs=True,
                    export_params=True)

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('model.onnx')"


def get_model():
    """Create model."""
    model = ImageCleanModel()
    return model


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Counter()

    model.train()

    criterion = fixed_loss()
    criterion = criterion.to(device)

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            noise_level_est, output = model(images)

            # out_image, gt_image, est_noise, gt_noise, if_asym
            loss = criterion(output, targets, noise_level_est, 0, 0)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.6f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            if os.environ["ENABLE_APEX"] == "YES":
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Counter()

    model.eval()

    criterion = fixed_loss()
    criterion = criterion.to(device)

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                noise_level_est, output = model(images)

            # out_image, gt_image, est_noise, gt_noise, if_asym
            loss = criterion(output, targets, noise_level_est, 0, 0)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            valid_loss.update(loss_value, count)
            t.set_postfix(loss='{:.6f}'.format(valid_loss.avg))
            t.update(count)


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Default environment variables avoid access exceptions
    if os.environ.get("ONLY_USE_CPU") != "YES" and os.environ.get("ONLY_USE_CPU") != "NO":
        os.environ["ONLY_USE_CPU"] = "NO"

    if os.environ.get("ENABLE_APEX") != "YES" and os.environ.get("ENABLE_APEX") != "NO":
        os.environ["ENABLE_APEX"] = "YES"

    if os.environ.get("DEVICE") != "YES" and os.environ.get("DEVICE") != "NO":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Is there GPU ?
    if not torch.cuda.is_available():
        os.environ["ONLY_USE_CPU"] = "YES"

    # export ONLY_USE_CPU=YES ?
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["ENABLE_APEX"] = "NO"
    else:
        try:
            from apex import amp
        except:
            os.environ["ENABLE_APEX"] = "NO"

    # Running on GPU if available
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["DEVICE"] = 'cpu'
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  USER: ", os.environ["USER"])
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])
    print("  ONLY_USE_CPU: ", os.environ["ONLY_USE_CPU"])
    print("  ENABLE_APEX: ", os.environ["ENABLE_APEX"])


def infer_perform():
    """Model infer performance ..."""

    model_setenv()
    device = os.environ["DEVICE"]

    model = ImageCleanModel()
    model.eval()
    model = model.to(device)

    with tqdm(total=len(1000)) as t:
        t.set_description(tag)

        # xxxx--modify here
        input = torch.randn(64, 3, 512, 512)
        input = input.to(device)

        with torch.no_grad():
            output = model(input)

        t.update(1)


if __name__ == '__main__':
    """Test model ..."""

    # model_export()
    # infer_perform()

    model = ImageCleanModel()
    print(model)
