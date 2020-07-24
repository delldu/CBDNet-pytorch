import torch
import torch.nn as nn
import torch.nn.functional as F

# The following class come from https://github.com/b03901165Shih/CBDNet-pytorch-inference.git
# Thanks !!!

class CBDNet(nn.Module):
    def __init__(self):
        super(CBDNet, self).__init__()
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

        x = self.US02_layer00(x)
        self.relu(x)
        x = self.US02_layer01(x)
        self.relu(x)
        x = self.US02_layer02(x)
        self.relu(x)
        x = self.USP01(x)
        x = torch.add(x, 1, down1_result)

        x = self.US01_layer00(x)
        self.relu(x)
        x = self.US01_layer01(x)
        self.relu(x)
        x = self.US01_layer02(x)
        y = torch.add(input, 1, x)
        del x, noise_level, down1_result, down2_result

        return y


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

        loss = torch.mean(torch.pow((out_image - gt_image), 2)) + \
                if_asym * 0.5 * torch.mean(torch.mul(torch.abs(0.3 - F.relu(gt_noise - est_noise)), torch.pow(est_noise - gt_noise, 2))) + \
                0.05 * tvloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

if __name__ == '__main__':
    model = CBDNet()
    print(model)
