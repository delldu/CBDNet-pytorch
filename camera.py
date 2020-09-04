import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
import pdb

torch.manual_seed(0) 


def CurveFit(x, y):
    n = torch.tensor(1.0, requires_grad=True)

    lr = 0.001
    for e in range(256):
        y_pred = torch.pow(x, n)

        loss = (y - y_pred).pow(2).sum()

        if(e % 10==0):
            print("Epoch: {} Loss: {}".format(e, loss.item()))

        loss.backward()
        with torch.no_grad():
            n -= lr * n.grad
            # Manually zero the gradients after updating weights
            n.grad.zero_()
    return n


class Debayer2x2(nn.Module):
    '''Demosaicing of Bayer images using 2x2 convolutions.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'.
    '''
    def __init__(self):
        super(Debayer2x2, self).__init__()

        self.kernels = nn.Parameter(
            torch.tensor([
                [1, 0],
                [0, 0],

                [0, 0.5],
                [0.5, 0],

                [0, 0],
                [0, 1],
            ]).view(3,1,2,2), requires_grad=False
        )

    def forward(self, x):
        '''Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        '''
        x = F.conv2d(x, self.kernels, stride=2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

class Camera(object):
    """Camera model."""

    def __init__(self):
        """Load 201 Camera Response Function Dataset."""

        self.CRF = torch.load("matdata/201_CRF_data.pth")
        CRF_para_file = "matdata/201_CRF_para.pth"
        if not os.path.exists(CRF_para_file):
            self.CRF_para = torch.Tensor(201)
            for index in range(201):
                print("Curve CRF fitting ", index, " ...")
                n = CurveFit(self.CRF['I'][index], self.CRF['B'][index])
                self.CRF_para[index] = n.detach()
            torch.save(self.CRF_para, CRF_para_file)
        else:
            self.CRF_para = torch.load(CRF_para_file)

    def brightness(self, tensor, index):
        """Get brightness from irradian."""
        return tensor.pow(self.CRF_para[index])

    def brightness_plot(self, index):
        """Check brightness fit curve."""
        print("index: ", index, "brightness parameter: ", self.CRF_para[index])
        x = self.CRF['I'][index]
        y = self.CRF['B'][index]
        y_pred = self.brightness(x, index)
        # Start draw
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(title='Brightness Curve', ylabel='Brightness', xlabel='Irradiance')
        ax.plot(x, y)
        ax.plot(x, y_pred, ':')
        ax.grid()
        plt.show()
                
    def irradiance(self, tensor, index):
        """Get irradiance from brightness."""
        n = 1.0/self.CRF_para[index]
        return tensor.pow(n)

    def irradiance_plot(self, index):
        """Check brightness fit curve."""
        print("index: ", index, "brightness parameter: ", self.CRF_para[index])
        x = self.CRF['B'][index]
        y = self.CRF['I'][index]
        y_pred = self.irradiance(x, index)
        # Start draw
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(title='Irradiance Curve', ylabel='Irradiance', xlabel='Brightness')
        ax.plot(x, y)
        ax.plot(x, y_pred, ':')
        ax.grid()
        plt.show()

    def encode_mosaic(self, rgb):
        """RGB to bayer."""

        c, h, w = rgb.size(0), rgb.size(1), rgb.size(2)

        bayer = torch.zeros((h, w))

        # Path is like N -- RGGB ?
        bayer[0:h:2, 0:w:2] = rgb[0, 0:h:2, 0:w:2]  # R
        bayer[1:h:2, 0:w:2] = rgb[1, 1:h:2, 0:w:2]  # G
        bayer[0:h:2, 1:w:2] = rgb[1, 0:h:2, 1:w:2]  # G
        bayer[1:h:2, 1:w:2] = rgb[2, 1:h:2, 1:w:2]  # B

        return bayer

    def decode_mosaic(self, bayer):
        """ Bayer --> RGB ."""
        model = Debayer2x2()
        input = bayer.unsqueeze(0).unsqueeze(0)
        output = model(input).squeeze(0)
        return output

    def makenoise(self, image):
        """Here image is CxHxW [0, 1.0] tensor and C==3."""

        c, h, w = image.size(0), image.size(1), image.size(2)

        sigma_s = torch.randn(3)
        sigma_s.uniform_(0.0, 0.16)

        sigma_c = torch.randn(3)
        sigma_c.uniform_(0.0, 0.06)

        index = random.randint(0, 200)

        # Camera light noise
        temp_x = self.irradiance(image, index)
        # signal noise
        noise_s = torch.zeros_like(temp_x)
        for ch in range(c):
            noise_s[ch, :, :] = sigma_s[ch] * temp_x[ch, :, :]
        noise_s = noise_s * torch.randn(noise_s.size())
        # random noise
        noise_c = torch.zeros_like(temp_x)
        for ch in range(c):
            noise_c[ch, :, :] = torch.normal(0, sigma_c[ch], (h, w))
        temp_x_n = temp_x + noise_s + noise_c
        temp_x = self.brightness(temp_x_n, index)

        # Camera sensor noise
        self.brightness_plot(index)
        temp_x.clamp_(0, 1.0)
        bayer = self.encode_mosaic(temp_x)
        noise_image = self.decode_mosaic(bayer)

        noise_image.clamp_(0, 1.0)

        return noise_image

    def noiselevel(self, t):
        pass

def TestCurveFit():
    camera = Camera()
    index = random.randint(0, 200)
    print("index == ", index)
    camera.brightness_plot(index)
    camera.irradiance_plot(index)

def TestMakeNoise():
    camera = Camera()

    # Make sure image size could be divied by 2 without resident
    image = Image.open("imgs/CBDNet_v13.png").convert("RGB")
    (w, h) = image.size
    w = (w // 2) * 2
    h = (h // 2) * 2
    if (w, h) != image.size:
        image = image.resize((w, h))

    image_tensor = transforms.ToTensor()(image)
    noise_image_tensor = camera.makenoise(image_tensor)
    noise_image = transforms.ToPILImage()(noise_image_tensor)
    noise_image.show()

    noise_level_tensor = noise_image_tensor - image_tensor
    noise_level = transforms.ToPILImage()(noise_level_tensor)
    noise_level.show()


# TestCurveFit()
TestMakeNoise()

