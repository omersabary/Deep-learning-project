

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def maybe_cuda(tensor, use_cuda):
    return tensor.cuda() if use_cuda else tensor

def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z ** 2, axis=1))[:, np.newaxis], 1)


def imsave(filename, array):
    im = Image.fromarray((array * 255).astype(np.uint8))
    im.save(filename)

class IndexedDataset(Dataset):
    """
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return (img, label, idx)

