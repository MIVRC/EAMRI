"""
some helper function related to fft and some torch operation
"""
import numpy as np
sqrt = np.sqrt
from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=False):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



def fftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(fft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def ifftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(ifft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def fft2c(x, norm="ortho"):
    '''
    Centered fft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(fft2(ifftshift(x, axes=axes), norm=norm), axes=axes)
    return res


def ifft2c(x, norm="ortho"):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(
        ifft2(ifftshift(x, axes=axes), norm=norm), axes=axes)
    return res


def fourier_matrix(rows, cols):
    '''
    parameters:
    rows: number or rows
    cols: number of columns
    return unitary (rows x cols) fourier matrix
    '''
    # from scipy.linalg import dft
    # return dft(rows,scale='sqrtn')


    col_range = np.arange(cols)
    row_range = np.arange(rows)
    scale = 1 / np.sqrt(cols)

    coeffs = np.outer(row_range, col_range)
    fourier_matrix = np.exp(coeffs * (-2. * np.pi * 1j / cols)) * scale

    return fourier_matrix


def inverse_fourier_matrix(rows, cols):
    return np.array(np.matrix(fourier_matrix(rows, cols)).getH())


def flip(m, axis):
    """
    ==== > Only in numpy 1.12 < =====
    Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.
    .. versionadded:: 1.12.0
    Parameters
    ----------
    m : array_like
        Input array.
    axis : integer
        Axis in array, which entries are reversed.
    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.
    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).
    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).
    flip(m, 1) is equivalent to fliplr(m).
    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.
    Examples
    --------
    >>> A = np.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> A = np.random.randn(3,4,5)
    >>> np.all(flip(A,2) == A[:,:,::-1,...])
    True
    """
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]


def rot90_nd(x, axes=(-2, -1), k=1):
    """Rotates selected axes"""
    def flipud(x):
        return flip(x, axes[0])

    def fliplr(x):
        return flip(x, axes[1])

    x = np.asanyarray(x)
    if x.ndim < 2:
        raise ValueError("Input must >= 2-d.")
    k = k % 4
    if k == 0:
        return x
    elif k == 1:
        return fliplr(x).swapaxes(*axes)
    elif k == 2:
        return fliplr(flipud(x))
    else:
        # k == 3
        return fliplr(x.swapaxes(*axes))

def torch_ifftshift(x):
    size = x.shape[-3:]
    indexs = []
    for i in range(2):
        n = size[i]
        p = n - (n + 1) // 2
        tmp = list(range(p, n))
        tmp.extend(list(range(0, p)))
        indexs.append(tmp)
    x = x[:, :, indexs[0]]
    x = x[:, :, :, indexs[1]]
    return x


def torch_fftshift(x):

    size = x.shape[-3:]
    indexs = []
    for i in range(2):
        n = size[i]
        p = (n + 1) // 2
        tmp = list(range(p, n))
        tmp.extend(list(range(0, p)))
        indexs.append(tmp)
    x = x[:, :, indexs[0]]
    x = x[:, :, :, indexs[1]]
    return x

def torch_fft2c(x):
    is_3d = False
    if x.dim() == 4:
        # 2D cnn.we need swap axes from [b, 2, h, w] -> [b, 1, h, w, 2]
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 4, 2)
    elif x.dim() == 5:
        x = x.permute(0, 1, 3, 4, 2)
        is_3d = True
    else:
        raise "Only 4 or 5 fimensions"

    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    y = torch_fftshift(torch.fft(torch_ifftshift(x), 2, normalized=True))
    if is_3d:
        return y.permute(0, 1, 4, 2, 3)
    else:
        y = y.squeeze(1)
        return y.permute(0, 3, 1, 2)


def torch_ifft2c(x):
    is_3d = False
    if x.dim() == 4:
        # 2D cnn.we need swap axes from [b, 2, h, w] -> [b, 1, h, w, 2]
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 4, 2)
    elif x.dim() == 5:
        x = x.permute(0, 1, 3, 4, 2)
        is_3d = True
    else:
        raise "Only 4 or 5 fimensions"
    y = torch_fftshift(torch.ifft(torch_ifftshift(x), 2, normalized=True))
    if is_3d:
        return y.permute(0, 1, 4, 2, 3)
    else:
        y = y.squeeze(1)
        return y.permute(0, 3, 1, 2)

