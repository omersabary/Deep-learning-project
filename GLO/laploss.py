import torch
from torch import nn
import torch.nn.functional as fnn
import numpy as np
from torch.autograd import Variable


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def build_gauss_kernel(self, size=5, sigma=1.0, n_channels=1, cuda=False):
        if size % 2 != 1:
            raise ValueError("kernel size must be uneven")
        grid = np.float32(np.mgrid[0:size, 0:size].T)
        gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
        kernel = np.sum(gaussian(grid), axis=2)
        kernel /= np.sum(kernel)
        # repeat same kernel across depth dimension
        kernel = np.tile(kernel, (n_channels, 1, 1))
        # conv weight should be (out_channels, groups/in_channels, h, w),
        # and since we have depth-separable convolution we want the groups dimension to be 1
        kernel = torch.FloatTensor(kernel[:, None, :, :])
        if cuda:
            kernel = kernel.cuda()
        return Variable(kernel, requires_grad=False)

    def conv_gauss(self, img, kernel):
        """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
        n_channels, _, kw, kh = kernel.shape
        img = fnn.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return fnn.conv2d(img, kernel, groups=n_channels)

    def laplacian_pyramid(self, img, kernel, max_levels=5):
        current = img
        pyr = []

        for level in range(max_levels):
            filtered = self.conv_gauss(current, kernel)
            diff = current - filtered
            pyr.append(diff)
            current = fnn.avg_pool2d(filtered, 2)

        pyr.append(current)
        return pyr

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = self.build_gauss_kernel(
                size=self.k_size, sigma=self.sigma,
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input = self.laplacian_pyramid( input, self._gauss_kernel, self.max_levels)
        pyr_target = self.laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
