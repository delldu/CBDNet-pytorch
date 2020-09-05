import torch
from torchvision import transforms
from PIL import Image

import pdb


def TensorNoiseLevel(t, PatchSize=8):
	'''
		input: t -- Image CxHxW tensor, [0, 1.0]
	'''
	step = 3
	C, H, W = t.size()
	PatchNumbers = len(range(0, H - PatchSize, step)) * len(range(0, W - PatchSize, step))
	PlainPatchSize = C * PatchSize * PatchSize

	# Collect patchs
	patchs = torch.zeros(PlainPatchSize, PatchNumbers)
	nn = 0
	for i in range(0, H - PatchSize, step):
		for j in range(0, W - PatchSize, step):
			patch = t[:, i : i + PatchSize, j : j + PatchSize]
			patchs[:, nn] = patch.reshape(PlainPatchSize)
			nn = nn + 1

	m = patchs.mean(dim = 1, keepdims=True)
	patchs = patchs - m

	matrix = torch.matmul(patchs, patchs.t())/PatchNumbers

	# [Real, Image] Eigen Values
	eigvals, _  = torch.eig(matrix)
	sigmas, _ = eigvals[0 : PlainPatchSize, 0].sort()

	# sigmas --- Real of egien values
	del patchs, matrix

	# Search noise level
	for i in range(PlainPatchSize, 0, -1):
		x = sigmas[:i]
		m = x.mean()
		left = (x < m).sum()
		right = (x > m).sum()
		if left == right:
			return m.sqrt()

	# Search failure ?
	return sigmas.median().sqrt()

def ImageNoiseLevel(image_filename):
	img = Image.open(image_filename).convert('RGB')
	t = transforms.ToTensor()(img)

	for i in [5, 15, 10, 25, 30, 40, 50, 60, 70, 80, 90, 100]:
		std = i / 255.0
		noise = t + std * torch.randn(t.size())
		sigma = TensorNoiseLevel(noise) * 255.0

		# if abs(i - sigma) > 1.0:
		print("{} Noise Level: Real is {:.4f}, Estimate is {:.4f}".format(image_filename, i, sigma))
		# if i % 5 == 0:
		# 	noise = noise.unsqueeze(0)
		# 	noise = transforms.ToPILImage()(noise)
		# 	noise.show()



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

	# ImageNoiseLevel("/tmp/bag.png")

	# ImageNoiseLevel("/tmp/test.jpg")	
	# ImageNoiseLevel("/tmp/lena.png")
