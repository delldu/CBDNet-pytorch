"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2020-2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 09月 09日 星期三 23:56:45 CST
# ***
# ************************************************************************************/
#

import torch
import torch.nn as nn

# https://github.com/b03901165Shih/CBDNet-pytorch-inference.git
class ImageCleanModel(nn.Module):
    """ImageClean Model."""

    def __init__(self):
        """Init model."""
        super(ImageCleanModel, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # Noise Estimation Network
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
        x = self.relu(self.E01(input))
        x = self.relu(self.E02(x))
        x = self.relu(self.E03(x))
        x = self.relu(self.E04(x))
        x = self.relu(self.E05(x))
        noise_level = x
        x = torch.cat((input, noise_level), dim=1)

        x = self.relu(self.DS01_layer00(x))
        x = self.relu(self.DS01_layer01(x))
        x = self.relu(self.DS01_layer02(x))
        x = self.relu(self.DS01_layer03(x))
        down1_result = x

        x = self.DS02(down1_result)
        x = self.DS02_layer00_cf(x)
        x = self.relu(self.DS02_layer00(x))
        x = self.relu(self.DS02_layer01(x))
        x = self.relu(self.DS02_layer02(x))
        down2_result = x

        x = self.DS03(down2_result)
        x = self.DS03_layer00_cf(x)
        x = self.relu(self.DS03_layer00(x))
        x = self.relu(self.DS03_layer01(x))
        x = self.relu(self.DS03_layer02(x))
        x = self.relu(self.UPS03_layer00(x))
        x = self.relu(self.UPS03_layer01(x))
        x = self.relu(self.UPS03_layer02(x))
        x = self.relu(self.UPS03_layer03(x))

        x = self.USP02(x)

        # x = torch.add(x, 1, down2_result)
        x = torch.add(x, down2_result, alpha=1)
        del down2_result

        x = self.relu(self.US02_layer00(x))
        x = self.relu(self.US02_layer01(x))
        x = self.relu(self.US02_layer02(x))
        x = self.USP01(x)
        # x = torch.add(x, 1, down1_result)
        x = torch.add(x, down1_result, alpha=1)
        del down1_result

        x = self.relu(self.US01_layer00(x))
        x = self.relu(self.US01_layer01(x))
        x = self.US01_layer02(x)
        # y = torch.add(input, 1, x)
        y = torch.add(input, x, alpha=1)
        del x

        return y.clamp(0.0, 1.0)
