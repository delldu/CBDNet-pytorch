import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pdb

def PCA_svd(X, k, center=True):
  n = X.size()[0]
  ones = torch.ones(n).view([n,1])
  h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
  H = torch.eye(n) - h
  H = H.cuda()
  X_center =  torch.mm(H.double(), X.double())
  u, s, v = torch.svd(X_center)
  pdb.set_trace()

  components  = v[:k].t()
  #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
  return components


PatchSize = 8

def TensorNoiseLevel(t):
	patch_set = []
	for i in range(t.size(0) - PatchSize):
		for j in range(t.size(1) - PatchSize):
			patch = t[i : i + PatchSize, j : j + PatchSize]
			# if (patch.std().item() < 0.1):
			patch_set.append(patch.reshape(PatchSize * PatchSize, 1))
	patchs = torch.cat(patch_set, 1)
	m = patchs.mean(1).reshape(PatchSize * PatchSize, 1)
	zero_mean_set = patchs - m
	nn = zero_mean_set.size(1)
	matrix = torch.zeros(PatchSize * PatchSize, 1)
	for i in range(nn):
		matrix = matrix + zero_mean_set[:, i : i + 1].mul(zero_mean_set[:, i : i + 1].t())
	matrix = matrix.div(nn)

	# matrix = zero_mean_set.mm(zero_mean_set.t())
	eigvals, _ = torch.eig(matrix)

	 # [Real, Image] Eigen Values
	sigmas, indices = eigvals[0 : PatchSize * PatchSize, 0].sort()

	# x = sigmas.numpy()
	# plt.plot(x,"r--o")
	# plt.show()

	x = max(sigmas.median(), 0.0).sqrt()
	# pdb.set_trace()

	return x.item()


def ImageNoiseLevel(image_filename):
	img = Image.open(image_filename).convert('L')
	t = transforms.ToTensor()(img)
	t = t.squeeze(0)
	for i in range(1, 100, 5):
		sigma = TensorNoiseLevel(t + i * torch.randn(t.size()))
		print("{} Noise Level: {:.4f} {:.4f}".format(image_filename, i, sigma))


if __name__ == '__main__':
	ImageNoiseLevel("test/input/01_noise.png")
	ImageNoiseLevel("/tmp/bag.png")
