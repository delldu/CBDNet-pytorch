"""Data loader."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 09月 01日 星期二 18:24:27 CST
# ***
# ************************************************************************************/
#

import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.utils as utils

PolyuRoot = 'dataset/Polyu/CroppedImages'


def get_transform(train=True):
    """Transform images."""
    ts = []
    # if train:
    #     ts.append(T.RandomHorizontalFlip(0.5))
    ts.append(T.ToTensor())
    return T.Compose(ts)

class PolyuDataset(data.Dataset):
    """Define dataset."""

    def __init__(self, root, transforms=get_transform()):
        """Init dataset."""
        super(PolyuDataset, self).__init__()

        self.root = root
        self.transforms = transforms

        # load all images, sorting for alignment
        files = list(sorted(os.listdir(root)))
        self.images = [fn for fn in files if fn.endswith('real.JPG')]

    def __getitem__(self, idx):
        """Load images."""
        image_path = os.path.join(self.root, self.images[idx])
        src = Image.open(image_path).convert("RGB")
        head, tail = os.path.split(image_path)
        image_path = os.path.join(head, tail.replace('real', 'mean'))
        tgt = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            src = self.transforms(src)
            tgt = self.transforms(tgt)
        return src, tgt

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)

    def __repr__(self):
        """
        Return printable representation of the dataset object.
        """
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def train_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""

    train_ds = PolyuDataset(PolyuRoot, get_transform(train=True))

    # Split train_ds in train and valid set
    valid_size = int(len(train_ds) * 0.20)      # 20%

    valid_len = int(0.2 * len(train_ds))
    indices = [i for i in range(valid_len, len(train_ds))]
    valid_ds = data.Subset(train_ds, indices)
    indices = [i for i in range(valid_len)]
    train_ds = data.Subset(train_ds, indices[:-valid_len])

    # Define training and validation data loaders
    train_dl = data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = data.DataLoader(valid_ds, batch_size=bs * 2, shuffle=False, num_workers=4)

    return train_dl, valid_dl

def test_data(bs):
    """Get data loader for test, bs means batch_size."""

    _, test_bs = train_data(bs)

    return test_dl


def get_data(trainning=True, bs=4):
    """Get data loader for trainning & validating, bs means batch_size."""

    return train_data(bs) if trainning else test_data(bs)


if __name__ == '__main__':
    ds = PolyuDataset(PolyuRoot)
    print(ds)
    src, tgt = ds[10]
    grid = utils.make_grid(torch.cat([src.unsqueeze(0), tgt.unsqueeze(0)], dim=0), nrow=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    image.show()
