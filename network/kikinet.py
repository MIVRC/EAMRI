# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Optional, List

import torch
import torch.nn as nn
from fastmri.data import transforms_simple as T
#from .didn import DIDN
from .mwcnn import MWCNN
from .networkUtil import *


#===============================================
# basic module
#===============================================


class Conv2d(nn.Module):
    """Implementation of a simple cascade of 2D convolutions.
    If `batchnorm` is set to True, batch normalization layer is applied after each convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_convs: int = 3,
        activation: nn.Module = nn.PReLU(),
        batchnorm: bool = False,
    ):
        """Inits :class:`Conv2d`.
        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        hidden_channels: int
            Number of hidden channels.
        n_convs: int
            Number of convolutional layers.
        activation: nn.Module
            Activation function.
        batchnorm: bool
            If True a batch normalization layer is applied after every convolution.
        """
        super().__init__()

        conv: List[nn.Module] = []
        for idx in range(n_convs):
            conv.append(
                nn.Conv2d(
                    in_channels if idx == 0 else hidden_channels,
                    hidden_channels if idx != n_convs - 1 else out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            if batchnorm:
                conv.append(nn.BatchNorm2d(hidden_channels if idx != n_convs - 1 else out_channels, eps=1e-4))
            if idx != n_convs - 1:
                conv.append(activation)
        self.conv = nn.Sequential(*conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of :class:`Conv2d`.
        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        Returns
        -------
        out: torch.Tensor
            Convoluted output.
        """
        out = self.conv(x)
        return out



class MultiCoil(nn.Module):
    """This makes the forward pass of multi-coil data of shape (N, N_coils, H, W, C) to a model.
    If coil_to_batch is set to True, coil dimension is moved to the batch dimension. Otherwise, it passes to the model
    each coil-data individually.
    """

    def __init__(self, model: nn.Module, coil_dim: int = 1, coil_to_batch: bool = False):
        """Inits :class:`MultiCoil`.
        Parameters
        ----------
        model: nn.Module
            Any nn.Module that takes as input with 4D data (N, H, W, C). Typically a convolutional-like model.
        coil_dim: int
            Coil dimension. Default: 1.
        coil_to_batch: bool
            If True batch and coil dimensions are merged when forwarded by the model and unmerged when outputted.
            Otherwise, input is forwarded to the model per coil.
        """
        super().__init__()

        self.model = model
        self.coil_to_batch = coil_to_batch
        self._coil_dim = coil_dim

    def _compute_model_per_coil(self, data: torch.Tensor) -> torch.Tensor:
        output = []

        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(self.model(subselected_data))

        return torch.stack(output, dim=self._coil_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of MultiCoil.
        Parameters
        ----------
        x: torch.Tensor
            Multi-coil input of shape (N, coil, height, width, in_channels).
        Returns
        -------
        out: torch.Tensor
            Multi-coil output of shape (N, coil, height, width, out_channels).
        """
        if self.coil_to_batch:
            x = x.clone()
            batch, coil, height, width, channels = x.size()

            x = x.reshape(batch * coil, height, width, channels).permute(0, 3, 1, 2).contiguous()
            x = self.model(x).permute(0, 2, 3, 1)
            x = x.reshape(batch, coil, height, width, -1)
        else:
            x = self._compute_model_per_coil(x).contiguous()

        return x



class ConvBlock(nn.Module):
    """U-Net convolutional block.
    It consists of two convolution layers each followed by instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_probability: float):
        """Inits ConvBlock.
        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of :class:`ConvBlock`.
        Parameters
        ----------
        input_data: torch.Tensor
        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        """Representation of :class:`ConvBlock`."""
        return (
            f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dropout_probability={self.dropout_probability})"
        )


class TransposeConvBlock(nn.Module):
    """U-Net Transpose Convolutional Block.
    It consists of one convolution transpose layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Inits :class:`TransposeConvBlock`.
        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`TransposeConvBlock`.
        Parameters
        ----------
        input_data: torch.Tensor
        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        """Representation of "class:`TransposeConvBlock`."""
        return f"ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels})"


class UnetModel2d(nn.Module):
    """PyTorch implementation of a U-Net model based on [1]_.
    References
    ----------
    .. [1] Ronneberger, Olaf, et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, edited by Nassir Navab et al., Springer International Publishing, 2015, pp. 234–41. Springer Link, https://doi.org/10.1007/978-3-319-24574-4_28.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
    ):
        """Inits :class:`UnetModel2d`.
        Parameters
        ----------
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_channels, num_filters, dropout_probability)])
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, dropout_probability)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, dropout_probability)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, dropout_probability)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, dropout_probability),
                nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1),
            )
        ]

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of :class:`UnetModel2d`.
        Parameters
        ----------
        input_data: torch.Tensor
        Returns
        -------
        torch.Tensor
        """
        stack = []
        output = input_data

        # Apply down-sampling layers
        for _, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/bottom if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)


        return output




class KIKINet(nn.Module):
    """Based on KIKINet implementation [1]_. Modified to work with multi-coil k-space data.
    References
    ----------
    .. [1] Eo, Taejoon, et al. “KIKI-Net: Cross-Domain Convolutional Neural Networks for Reconstructing Undersampled Magnetic Resonance Images.” Magnetic Resonance in Medicine, vol. 80, no. 5, Nov. 2018, pp. 2188–201. PubMed, https://doi.org/10.1002/mrm.27201.
    """

    def __init__(
        self,
        image_model_architecture: str = "MWCNN",
        kspace_model_architecture: str = "DIDN",
        num_iter: int = 10,
        shift: bool = False,
        **kwargs,
    ):
        """Inits :class:`KIKINet`.
        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        image_model_architecture: str
            Image model architecture. Currently only implemented for MWCNN and (NORM)UNET. Default: 'MWCNN'.
        kspace_model_architecture: str
            Kspace model architecture. Currently only implemented for CONV and DIDN and (NORM)UNET. Default: 'DIDN'.
        num_iter: int
            Number of unrolled iterations.
        normalize: bool
            If true, input is normalised based on input scaling_factor.
        kwargs: dict
            Keyword arguments for model architectures.
        """
        super().__init__()
        image_model: nn.Module
        if image_model_architecture == "MWCNN":
            image_model = MWCNN(
                input_channels=2,
                first_conv_hidden_channels=kwargs.get("image_mwcnn_hidden_channels", 32),
                num_scales=kwargs.get("image_mwcnn_num_scales", 4),
                bias=kwargs.get("image_mwcnn_bias", False),
                batchnorm=kwargs.get("image_mwcnn_batchnorm", False),
            )
        elif image_model_architecture in ["UNET", "NORMUNET"]:
            unet = UnetModel2d if image_model_architecture == "UNET" else NormUnetModel2d
            image_model = unet(
                in_channels=2,
                out_channels=2,
                num_filters=kwargs.get("image_unet_num_filters", 8),
                num_pool_layers=kwargs.get("image_unet_num_pool_layers", 4),
                dropout_probability=kwargs.get("image_unet_dropout_probability", 0.0),
            )
        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented only with image_model_architecture == 'MWCNN', 'UNET' or 'NORMUNET."
                f"Got {image_model_architecture}."
            )

        kspace_model: nn.Module
        if kspace_model_architecture == "CONV":
            kspace_model = Conv2d(
                in_channels=2,
                out_channels=2,
                hidden_channels=kwargs.get("kspace_conv_hidden_channels", 16),
                n_convs=kwargs.get("kspace_conv_n_convs", 4),
                batchnorm=kwargs.get("kspace_conv_batchnorm", False),
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model = DIDN(
                in_channels=2,
                out_channels=2,
                hidden_channels=kwargs.get("kspace_didn_hidden_channels", 16),
                num_dubs=kwargs.get("kspace_didn_num_dubs", 6),
                num_convs_recon=kwargs.get("kspace_didn_num_convs_recon", 9),
            )
        elif kspace_model_architecture in ["UNET", "NORMUNET"]:
            unet = UnetModel2d if kspace_model_architecture == "UNET" else NormUnetModel2d
            kspace_model = UnetModel2d(
                in_channels=2,
                out_channels=2,
                num_filters=kwargs.get("kspace_unet_num_filters", 8),
                num_pool_layers=kwargs.get("kspace_unet_num_pool_layers", 4),
                dropout_probability=kwargs.get("kspace_unet_dropout_probability", 0.0),
            )
        else:
            raise NotImplementedError(
                f"XPDNet is currently implemented for kspace_model_architecture == 'CONV', 'DIDN',"
                f" 'UNET' or 'NORMUNET'. Got kspace_model_architecture == {kspace_model_architecture}."
            )

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

        self.image_model_list = nn.ModuleList([image_model] * num_iter)
        self.kspace_model_list = nn.ModuleList([MultiCoil(kspace_model, self._coil_dim)] * num_iter)

        self.sens_net = SensitivityModel(chans = 8, num_pools = 4, shift=shift)

        self.num_iter = num_iter
        self.shift = shift

    def forward(
        self,
        zf: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        scaling_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`KIKINet`.
        Parameters
        ----------
        zf: torch.Tensor
            zero-filled image (N, coil, height, width, complex= 2), do not use
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        scaling_factor: Optional[torch.Tensor]
            Scaling factor of shape (N,). If None, no scaling is applied. Default: None.
        Returns
        -------
        image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """

        sensitivity_map = self.sens_net(masked_kspace, sampling_mask)

        kspace = masked_kspace.clone()

        '''
        if self.normalize and scaling_factor is not None:
            kspace = kspace / (scaling_factor**2).view(-1, 1, 1, 1, 1)
        '''

        for idx in range(self.num_iter):
            tmp = kspace.permute(0, 1, 4, 2, 3)
            tmp1 = self.kspace_model_list[idx](tmp)
            kspace = tmp1.permute(0, 1, 3, 4, 2)

            # reduce
            image = T.reduce_operator(
                T.ifft2(
                    torch.where(
                        sampling_mask == 0,
                        torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device),
                        kspace,
                    ),
                    shift = self.shift
                ),
                sensitivity_map,
                self._coil_dim,
            )

            image = self.image_model_list[idx](image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) #(B, H, W, 2)

            # dc
            if idx < self.num_iter - 1:
                kspace = torch.where(
                    sampling_mask == 0,
                    torch.tensor([0.0], dtype=image.dtype).to(image.device),
                    T.fft2(
                        T.expand_operator(image, sensitivity_map, self._coil_dim), shift=self.shift),
                )
       
        return image.permute(0,3,1,2)
