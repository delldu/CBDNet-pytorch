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

	return max(sigmas.median(), 0.0).sqrt().item()


def ImageNoiseLevel(image_filename):
	img = Image.open(image_filename).convert('L')
	t = transforms.ToTensor()(img)
	t = t.squeeze(0)
	for i in range(1, 100):
		sigma = TensorNoiseLevel(t + i * torch.randn(t.size()))
		print("{} Noise Level: {:.4f} {:.4f}".format(image_filename, i, sigma))


if __name__ == '__main__':
	ImageNoiseLevel("test/input/01_noise.png")
	ImageNoiseLevel("/tmp/bag.png")
