import torch
from torchvision import transforms
from PIL import Image

# import pdb


def TensorNoiseLevel(t, PatchSize=32, use_GPU=False):
	'''
		input: t -- 2-dimensions tensor
	'''
	nn = 0
	matrix = torch.zeros(PatchSize, PatchSize)

	if use_GPU:
		t = t.cuda()
		matrix = matrix.cuda()

	step = max(PatchSize//2, 1)
	for i in range(0, t.size(0) - PatchSize, step):
		for j in range(0, t.size(1) - PatchSize, step):
			patch = t[i : i + PatchSize, j : j + PatchSize]
			patch = patch - patch.mean()
			matrix = matrix + patch.mul(patch.t())
			nn = nn + 1
	matrix = matrix.div(nn)

	eigvals, _ = torch.eig(matrix)

	# [Real, Image] Eigen Values
	sigmas, _ = eigvals[0 : PatchSize * PatchSize, 0].sort()

	# Calibration ...
	output = sigmas.clamp(0.0, 100.0)
	output = output.sqrt()
	output = output * 255.0
	x = output[PatchSize//4].item()
	index = PatchSize//2
	if x < 5.0:
		index = PatchSize//32
	if x < 10.0:
		index = PatchSize//16
	if x < 20.0:
		index = PatchSize//8
	if x < 40.0:
		index = PatchSize//4
	if x < 80.0:
		index = PatchSize//2
	# x >= 80.0, ==> index == PatchSize // 2
	x = output[index].item()
	y = x - 1.0
	if x < 40.0:
		y = y - 1.0
	if x < 20.0:
		y = y - 2.0
	if x < 10.0:
		y = y - 3.0

	return max(y, 0.0)


def ImageNoiseLevel(image_filename):
	img = Image.open(image_filename).convert('L')
	t = transforms.ToTensor()(img)
	t = t.squeeze(0)
	for i in range(0, 90, 5):
		std = i / 255.0
		noise = t + std * torch.randn(t.size())
		sigma = TensorNoiseLevel(noise)

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
