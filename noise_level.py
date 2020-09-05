import torch
from torchvision import transforms
from PIL import Image

import pdb


def TensorNoiseLevel(t, patch_size=8):
	'''
		input: t -- Image CxHxW tensor, [0, 1.0]
	'''
	C, H, W = t.size()
	stride = patch_size // 2 + 1
	total_patch_size = C * patch_size * patch_size

	# Collect patchs
	num_patchs = len(range(0, H - patch_size + 1, stride)) * len(range(0, W - patch_size + 1, stride))
	patchs = torch.zeros(C, patch_size, patch_size, num_patchs)
	k = 0
	for i in range(0, H - patch_size + 1, stride):
		for j in range(0, W - patch_size + 1, stride):
			patchs[:, :, :, k] = t[:, i : i + patch_size, j : j + patch_size]
			k += 1

	patchs = patchs.reshape(total_patch_size, num_patchs)
	m = patchs.mean(dim = 1, keepdims=True)
	patchs = patchs - m
	matrix = torch.matmul(patchs, patchs.t())/num_patchs

	# [Real, Image] Eigen Values
	eigvals, _  = torch.eig(matrix)
	sigmas, _ = eigvals[0 : total_patch_size, 0].sort()
	# Now sigmas is real part of egien values
	del patchs, matrix

	# Search noise level
	for i in range(total_patch_size, 0, -1):
		x = sigmas[:i]
		m = sigmas[:i].mean()
		left = (x < m).sum()
		right = (x > m).sum()
		if left == right:
			break
	return m.sqrt()

def ImageNoiseLevel(image_filename):
	img = Image.open(image_filename).convert('RGB')

	t = transforms.ToTensor()(img)

	for i in [5, 15, 10, 25, 30, 40, 50, 60, 70, 80, 90, 100]:
		std = i / 255.0
		noise = t + std * torch.randn(t.size())
		sigma = TensorNoiseLevel(noise) * 255.0

		print("{} Noise Level: Real is {:.4f}, Estimation is {:.4f}".format(image_filename, i, sigma))

if __name__ == '__main__':
	ImageNoiseLevel("test/output/01_noise.png")
	ImageNoiseLevel("test/output/02_noise.png")
	ImageNoiseLevel("test/output/03_noise.png")
	ImageNoiseLevel("test/output/04_noise.png")
	ImageNoiseLevel("test/output/05_noise.png")
	ImageNoiseLevel("test/output/06_noise.png")
	ImageNoiseLevel("test/output/07_noise.png")
	ImageNoiseLevel("test/output/08_noise.png")
	ImageNoiseLevel("test/output/09_noise.png")

	# ImageNoiseLevel("/tmp/lena.png")
	# ImageNoiseLevel("/tmp/test.jpg")
