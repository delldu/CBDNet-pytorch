import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import pdb


class GaussFilter(nn.Module):
    """
    3x3 Guassian filter
    """

    def __init__(self):
        super(GaussFilter, self).__init__()
        self.conv = nn.Conv2d(
            3, 3, kernel_size=3, padding=1, groups=3, bias=False)

        # self.conv.bias.data.fill_(0.0)
        self.conv.weight.data.fill_(0.0625)
        self.conv.weight.data[:, :, 0, 1] = 0.125
        self.conv.weight.data[:, :, 1, 0] = 0.125
        self.conv.weight.data[:, :, 1, 2] = 0.125
        self.conv.weight.data[:, :, 2, 1] = 0.125
        self.conv.weight.data[:, :, 1, 1] = 0.25

    def forward(self, x):
        y = x.unsqueeze(0)
        y = self.conv(y)
        return y.squeeze(0)


class LaplaceFilter(nn.Module):
    """
    3x3 Guassian filter
    """

    def __init__(self):
        super(LaplaceFilter, self).__init__()
        self.conv = nn.Conv2d(
            3, 3, kernel_size=3, padding=1, groups=3, bias=False)

        # self.conv.bias.data.fill_(0.0)
        self.conv.weight.data.fill_(0.25)
        self.conv.weight.data[:, :, 0, 1] = -0.50
        self.conv.weight.data[:, :, 1, 0] = -0.50
        self.conv.weight.data[:, :, 1, 2] = -0.50
        self.conv.weight.data[:, :, 2, 1] = -0.50
        self.conv.weight.data[:, :, 1, 1] = 1.0

    def forward(self, x):
        y = x.unsqueeze(0)
        y = self.conv(y)
        return y.squeeze(0)


def TensorNoiseLevel(t, patch_size=8):
	'''
		input: t -- Image CxHxW tensor, [0, 1.0]
	'''
	C, H, W = t.size()
	stride = patch_size //2 + 1
	total_patch_size = C * patch_size * patch_size

	# Generate dataset
	num_patchs = len(range(0, H - patch_size + 1, stride)) * len(range(0, W - patch_size + 1, stride))
	patchs = torch.zeros(C, patch_size, patch_size, num_patchs)
	k = 0
	for i in range(0, H - patch_size + 1, stride):
		for j in range(0, W - patch_size + 1, stride):
			patchs[:, :, :, k] = t[:, i : i + patch_size, j : j + patch_size]
			k += 1
	patchs = patchs.reshape(total_patch_size, num_patchs)

	# 2. Create covariance matrix
	m = patchs.mean(dim = 1, keepdims=True)
	patchs = patchs - m
	matrix = torch.matmul(patchs, patchs.t())/num_patchs

	# 3. Calulate eigen value
	# [Real, Image] Eigen Values
	eigvals, _  = torch.eig(matrix)
	sigmas, _ = eigvals[0 : total_patch_size, 0].sort(descending=True)
	# Now sigmas is real part of egien values
	del patchs, matrix

	# 4. Search median value
	for i in range(total_patch_size):
		x = sigmas[i:]
		m = x.mean()
		left = (x < m).sum()
		right = (x > m).sum()
		if left == right:
			break
	return m.sqrt()*255.0

def ImageNoiseLevel(image_filename):
	img = Image.open(image_filename).convert('RGB')

	t = transforms.ToTensor()(img)
	filter = LaplaceFilter()

	# gt = filter(GaussFilter()(t))
	sigma = TensorNoiseLevel(t)
	laplace = filter(t)
	laplace_sigma = laplace.abs().mean()*53.25*4.0
	# torch.sqrt(3.14/72.0)*255.0

	print("Raw image {} noise estimation is {:.4f}, laplace: {:.4f}".format(image_filename, sigma, laplace_sigma))

	for i in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]:
		std = i / 255.0
		noise_c = std * torch.randn(t.size())
		noise = t + noise_c

		mask1 = noise >= 0.0
		mask2 = noise <= 1.0
		noise_c *= mask1 * mask2

		real_std = noise_c.std() * 255.0
		noise.clamp_(0.0, 1.0)

		# noise = filter(GaussFilter()(noise))
		if i == 100:
			laplace_image = transforms.ToPILImage()(noise)
			laplace_image.show()

		sigma = TensorNoiseLevel(noise)
		laplace = filter(noise)
		laplace_sigma = laplace.abs().mean()*53.25*4.0

		print("{}: {} Noise Level: Real is {:.4f}, Estimation is {:.4f}, {:.4f}".format(i, image_filename, real_std, sigma, laplace_sigma))
		# if i == 100:
		# 	noise_image = transforms.ToPILImage()(noise)
		# 	noise_image.show()

if __name__ == '__main__':
	ImageNoiseLevel("test/input/01_noise.png")
	ImageNoiseLevel("test/input/02_noise.png")
	# ImageNoiseLevel("test/input/03_noise.png")
	# ImageNoiseLevel("test/input/04_noise.png")
	# ImageNoiseLevel("test/input/05_noise.png")
	# ImageNoiseLevel("test/input/06_noise.png")
	# ImageNoiseLevel("test/input/07_noise.png")
	# ImageNoiseLevel("test/input/08_noise.png")
	# ImageNoiseLevel("test/input/09_noise.png")

	# ImageNoiseLevel("/tmp/lena.png")
	# ImageNoiseLevel("/tmp/test.png")

	# ImageNoiseLevel(sys.argv[1])
