"""
# adapated from fastmri

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import pdb
from .subsample import MaskFunc


try: #<= pytorch 1.7.0
    fft = torch.fft
    ifft = torch.ifft   
    rfft = torch.rfft
    irfft = torch.irfft

except AttributeError: #>= pytorch 1.8.0
    # Forwards compatibility for new pytorch versions
    # signal_ndim is compatible with pt1.7 
    def fft(input, signal_ndim, normalized=True):
        return torch.view_as_real(torch.fft.fft2(
            torch.view_as_complex(input),
            norm="ortho" if normalized else "backward"))
            
    def ifft(input, signal_ndim, normalized=True):
        return torch.view_as_real(torch.fft.ifft2(
            torch.view_as_complex(input),
            norm="ortho" if normalized else "backward"))


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1).sqrt()


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)



def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def ifft2(data, normalized=True, shift=True):
    """
    data: (H, W, 2) or (B, H, W, 2)
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    if shift:
        data = ifftshift(data, dim=[-3, -2])
    data = ifft(data, 2, normalized=normalized)
    if shift:
        data = fftshift(data, dim=[-3, -2])

    return data


def fft2(data, normalized=True, shift=True):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    if shift:
        data = ifftshift(data, dim=(-3, -2))
    data = fft(data, 2, normalized=normalized)
    if shift:
        data = fftshift(data, dim=(-3, -2))
    return data


def rfft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    data = ifftshift(data, dim=(-2, -1))
    data = torch.rfft(data, 2, normalized=True, onesided=False)
    data = fftshift(data, dim=(-3, -2))
    return data

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values). #(bs, h, w, channels)
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    return masked_data, mask




def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def center_crop_kspace(kspace, shape, size = [320,320], shift=True):
    
    assert kspace.shape[-1] == 2, 'invalid input shape for kspace'
    img = ifft2(kspace, shift=shift)
    w_from = (shape[0] - size[0]) // 2
    h_from = (shape[1] - size[1]) // 2
    w_to = w_from + size[0]
    h_to = h_from + size[1]
    img_crop = img[w_from:w_to, h_from:h_to,:]
    kspace_crop = fft2(img_crop)
    
    return kspace_crop





def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std





def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


#========================================================

def safe_divide(input_tensor, other_tensor): 
    """Divide input_tensor and other_tensor safely, set the output to zero where the divisor b is zero.
    Parameters
    ----------
    input_tensor: torch.Tensor, numerator
    other_tensor: torch.Tensor, dinominator
    Returns
    -------
    torch.Tensor: the division.
    """
    
    data = torch.where(
        other_tensor == 0,
        torch.tensor([0.0], dtype=input_tensor.dtype),
        input_tensor / other_tensor,
    )
    return data





def circular_centered_mask(shape, radius):
    center = np.asarray(shape) // 2
    Y, X = np.ogrid[: shape[0], : shape[1]]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    mask = ((dist_from_center <= radius) * np.ones(shape)).astype(bool)
    return mask[np.newaxis, ..., np.newaxis]



def cal_acs_mask(shape, radius=18):
    """
    calculate autocalibration mask
    shape: [height, width, 1]
    """
    assert len(shape) == 2, "dev: invalid input shape for cal_acs_mask"

    return torch.from_numpy(circular_centered_mask(shape,radius))


class EstimateSensitivityMap():
    """Data Transformer for training MRI reconstruction models.
    Estimates sensitivity maps given kspace data.
    modified from Direct
    """

    def __init__(
        self,
        type_of_map="rss_estimate",
        gaussian_sigma= None,
        shift= False,
        ): 
        """Inits :class:`EstimateSensitivityMap`.
        Parameters
        ----------
            K-space key. Default `kspace`.
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        type_of_map: str, optional
            Type of map to estimate. Can be "unit" or "rss_estimate". Default: "unit".
        gaussian_sigma: float, optional
            If non-zero, acs_image well be calculated
        """
        super().__init__()
        if type_of_map not in ["unit", "rss_estimate"]:
            raise ValueError(f"Expected type of map to be either `unit` or `rss_estimate`. Got {type_of_map}.")
        self.type_of_map = type_of_map
        self.gaussian_sigma = gaussian_sigma
        self.shift = shift 

    def estimate_acs_image(self, kspace_data, width_dim=-2):
        """Estimates the autocalibration (ACS) image by sampling the k-space using the ACS mask.
        Parameters
        ----------
        kspace_data: (coil, [slice], height, width, complex=2)

        width_dim: int
            Dimension corresponding to width. Default: -2.
        Returns
        -------
        acs_image: torch.Tensor
            Estimate of the ACS image.
        """
        assert kspace_data.shape[-1] == 2, "estimate sensitivity map, the last dimension of input kspace should be 2"

        kspace_shape = kspace_data.shape[1:-1] #([slices], h, w, 2)
        acs_mask = cal_acs_mask(kspace_shape)

        if self.gaussian_sigma == 0 or not self.gaussian_sigma:
            kspace_acs = kspace_data * acs_mask + 0.0  # + 0.0 removes the sign of zeros.
        else:
            gaussian_mask = torch.linspace(-1, 1, kspace_data.size(width_dim), dtype=kspace_data.dtype)
            gaussian_mask = torch.exp(-((gaussian_mask / self.gaussian_sigma) ** 2))
            gaussian_mask_shape = torch.ones(len(kspace_data.shape)).int()
            gaussian_mask_shape[width_dim] = kspace_data.size(width_dim)
            gaussian_mask = gaussian_mask.reshape(tuple(gaussian_mask_shape))
            kspace_acs = kspace_data * acs_mask * gaussian_mask + 0.0

        # Get complex-valued data solution
        acs_image = ifft2(kspace_acs, shift=self.shift) # Shape (coil, [slice], height, width, complex=2)

        return acs_image

    def __call__(self, kspace, coil_dim= 0): 
        """Calculates sensitivity maps for the input sample.
        Parameters
        ----------
        sample: Dict[str, Any]
            Must contain key matching kspace_key with value a (complex) torch.Tensor
            of shape (coil, height, width, complex=2).
        coil_dim: int
            Coil dimension. Default: 0.
        Returns
        -------
        sample: Dict[str, Any]
            Sample with key "sensitivity_map" with value the estimated sensitivity map.
        """
        if self.type_of_map == "unit":
            
            sensitivity_map = torch.zeros(kspace.shape).float()

            # Assumes complex channel is last
            sensitivity_map[..., 0] = 1.0  # Shape (coil, [slice], height, width, complex=2)


        elif self.type_of_map == "rss_estimate":
            acs_image = self.estimate_acs_image(kspace) # Shape (coil, [slice], height, width, complex=2)
            acs_image_abs = root_sum_of_squares(acs_image, dim=3) #(coil, [slice], height, width)
            acs_image_rss = root_sum_of_squares(acs_image_abs, dim=coil_dim) # Shape ([slice], height, width)
            acs_image_rss = acs_image_rss.unsqueeze(0).unsqueeze(-1) # Shape (1, [slice], height, width, 1)
            sensitivity_map = safe_divide(acs_image, acs_image_rss) # Shape (coil, [slice], height, width, complex=2)

        return sensitivity_map



def conjugate(data): 
    """Compute the complex conjugate of a torch tensor where the last axis denotes the real and complex part (last axis has dimension 2).
    Parameters
    ----------
    data: torch.Tensor
    Returns
    -------
    conjugate_tensor: torch.Tensor
    """
    assert data.shape[-1] == 2, 'input data should have last dimension 2'
    data = data.clone()  # Clone is required as the data in the next line is changed in-place.
    data[..., 1] = data[..., 1] * -1.0

    return data



def complex_mul(input_tensor, other_tensor):
    """Multiplies two complex-valued tensors. Assumes input tensors are complex (last axis has dimension 2).
    Parameters
    ----------
    input_tensor: torch.Tensor
        Input data
    other_tensor: torch.Tensor
        Input data
    Returns
    -------
    torch.Tensor
    """
    #assert_complex(input_tensor, complex_last=True)
    #assert_complex(other_tensor, complex_last=True)

    assert input_tensor.shape[-1] == 2, "complex_mul, input tensor should have last dimension 2"

    complex_index = -1

    real_part = input_tensor[..., 0] * other_tensor[..., 0] - input_tensor[..., 1] * other_tensor[..., 1]
    imaginary_part = input_tensor[..., 0] * other_tensor[..., 1] + input_tensor[..., 1] * other_tensor[..., 0]

    multiplication = torch.cat(
        [
            real_part.unsqueeze(dim=complex_index),
            imaginary_part.unsqueeze(dim=complex_index),
        ],
        dim=complex_index,
    )

    return multiplication


def reduce_operator(
    coil_data,
    sensitivity_map,
    dim= 0,
    ):
    r"""
    Given zero-filled reconstructions from multiple coils :math:`\{x_i\}_{i=1}^{N_c}` and
    coil sensitivity maps :math:`\{S_i\}_{i=1}^{N_c}` it returns:
        .. math::
            R(x_{1}, .., x_{N_c}, S_1, .., S_{N_c}) = \sum_{i=1}^{N_c} {S_i}^{*} \times x_i.
    Adapted from [1]_.
    Parameters
    ----------
    coil_data: torch.Tensor
        Zero-filled reconstructions from coils. Should be a complex tensor (with complex dim of size 2).
    sensitivity_map: torch.Tensor
        Coil sensitivity maps. Should be complex tensor (with complex dim of size 2).
    dim: int
        Coil dimension. Default: 0.
    Returns
    -------
    torch.Tensor:
        Combined individual coil images.
    References
    ----------
    .. [1] Sriram, Anuroop, et al. “End-to-End Variational Networks for Accelerated MRI Reconstruction.” ArXiv:2004.06688 [Cs, Eess], Apr. 2020. arXiv.org, http://arxiv.org/abs/2004.06688.
    """
    return complex_mul(conjugate(sensitivity_map), coil_data).sum(dim)


def expand_operator(
    data,
    sensitivity_map,
    dim = 0,
):
    r"""
    Given a reconstructed image :math:`x` and coil sensitivity maps :math:`\{S_i\}_{i=1}^{N_c}`, it returns
        .. math::
            E(x) = (S_1 \times x, .., S_{N_c} \times x) = (x_1, .., x_{N_c}).
    Adapted from [1]_.
    Parameters
    ----------
    data: torch.Tensor
        Image data. Should be a complex tensor (with complex dim of size 2).
    sensitivity_map: torch.Tensor
        Coil sensitivity maps. Should be complex tensor (with complex dim of size 2).
    dim: int
        Coil dimension. Default: 0.
    Returns
    -------
    torch.Tensor:
        Zero-filled reconstructions from each coil.
    References
    ----------
    .. [1] Sriram, Anuroop, et al. “End-to-End Variational Networks for Accelerated MRI Reconstruction.” ArXiv:2004.06688 [Cs, Eess], Apr. 2020. arXiv.org, http://arxiv.org/abs/2004.06688.
    """
    return complex_mul(sensitivity_map, data.unsqueeze(dim))
