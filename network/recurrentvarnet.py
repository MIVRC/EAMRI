# Copyright (c) DIRECT Contributors

from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms_simple as T
from .networkUtil import *


#===============================================
# basic module
#===============================================

class Conv2dGRU(nn.Module):
    """2D Convolutional GRU Network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        orthogonal_initialization: bool = True,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
    ):
        """Inits :class:`Conv2dGRU`.
        Parameters
        ----------
        in_channels: int
            Number of input channels.
        hidden_channels: int
            Number of hidden channels.
        out_channels: Optional[int]
            Number of output channels. If None, same as in_channels. Default: None.
        num_layers: int
            Number of layers. Default: 2.
        gru_kernel_size: int
            Size of the GRU kernel. Default: 1.
        orthogonal_initialization: bool
            Orthogonal initialization is used if set to True. Default: True.
        instance_norm: bool
            Instance norm is used if set to True. Default: False.
        dense_connect: int
            Number of dense connections.
        replication_padding: bool
            If set to true replication padding is applied.
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dense_connect = dense_connect

        self.reset_gates = nn.ModuleList([])
        self.update_gates = nn.ModuleList([])
        self.out_gates = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Create convolutional blocks
        for idx in range(num_layers + 1):
            in_ch = in_channels if idx == 0 else (1 + min(idx, dense_connect)) * hidden_channels
            out_ch = hidden_channels if idx < num_layers else out_channels
            padding = 0 if replication_padding else (2 if idx == 0 else 1)
            block: List[nn.Module] = []
            if replication_padding:
                if idx == 1:
                    block.append(nn.ReplicationPad2d(2))
                else:
                    block.append(nn.ReplicationPad2d(2 if idx == 0 else 1))
            block.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=5 if idx == 0 else 3,
                    dilation=(2 if idx == 1 else 1),
                    padding=padding,
                )
            )
            self.conv_blocks.append(nn.Sequential(*block))

        # Create GRU blocks
        for idx in range(num_layers):
            for gru_part in [self.reset_gates, self.update_gates, self.out_gates]:
                gru_block: List[nn.Module] = []
                if instance_norm:
                    gru_block.append(nn.InstanceNorm2d(2 * hidden_channels))
                gru_block.append(
                    nn.Conv2d(
                        in_channels=2 * hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=gru_kernel_size,
                        padding=gru_kernel_size // 2,
                    )
                )
                gru_part.append(nn.Sequential(*gru_block))

        if orthogonal_initialization:
            for reset_gate, update_gate, out_gate in zip(self.reset_gates, self.update_gates, self.out_gates):
                nn.init.orthogonal_(reset_gate[-1].weight)
                nn.init.orthogonal_(update_gate[-1].weight)
                nn.init.orthogonal_(out_gate[-1].weight)
                nn.init.constant_(reset_gate[-1].bias, -1.0)
                nn.init.constant_(update_gate[-1].bias, 0.0)
                nn.init.constant_(out_gate[-1].bias, 0.0)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Conv2dGRU forward pass given tensors `cell_input` and `previous_state`.
        Parameters
        ----------
        cell_input: torch.Tensor
            Input tensor.
        previous_state: torch.Tensor
            Tensor of previous hidden state.
        Returns
        -------
        out, new_states: (torch.Tensor, torch.Tensor)
            Output and new states.
        """
        new_states: List[torch.Tensor] = []
        conv_skip: List[torch.Tensor] = []

        if previous_state is None:
            batch_size, spatial_size = cell_input.size(0), (cell_input.size(2), cell_input.size(3))
            state_size = [batch_size, self.hidden_channels] + list(spatial_size) + [self.num_layers]
            previous_state = torch.zeros(*state_size, dtype=cell_input.dtype).to(cell_input.device)

        for idx in range(self.num_layers):
            if len(conv_skip) > 0:
                cell_input = F.relu(
                    self.conv_blocks[idx](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1)),
                    inplace=True,
                )
            else:
                cell_input = F.relu(self.conv_blocks[idx](cell_input), inplace=True)
            if self.dense_connect > 0:
                conv_skip.append(cell_input)

            prev_state = previous_state[:, :, :, :, idx]
            stacked_inputs = torch.cat([cell_input, prev_state], dim=1)

            update = torch.sigmoid(self.update_gates[idx](stacked_inputs))
            reset = torch.sigmoid(self.reset_gates[idx](stacked_inputs))
            delta = torch.tanh(self.out_gates[idx](torch.cat([cell_input, prev_state * reset], dim=1)))
            cell_input = prev_state * (1 - update) + delta * update
            new_states.append(cell_input)
            cell_input = F.relu(cell_input, inplace=False)
        if len(conv_skip) > 0:
            out = self.conv_blocks[self.num_layers](torch.cat([*conv_skip[-self.dense_connect :], cell_input], dim=1))
        else:
            out = self.conv_blocks[self.num_layers](cell_input)

        return out, torch.stack(new_states, dim=-1)


class NormConv2dGRU(nn.Module):
    """Normalized 2D Convolutional GRU Network.
    Normalization methods adapted from NormUnet of [1]_.
    References
    ----------
    .. [1] https://github.com/facebookresearch/fastMRI/blob/
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        gru_kernel_size=1,
        orthogonal_initialization: bool = True,
        instance_norm: bool = False,
        dense_connect: int = 0,
        replication_padding: bool = True,
        norm_groups: int = 2,
    ):
        """Inits :class:`NormConv2dGRU`.
        Parameters
        ----------
        in_channels: int
            Number of input channels.
        hidden_channels: int
            Number of hidden channels.
        out_channels: Optional[int]
            Number of output channels. If None, same as in_channels. Default: None.
        num_layers: int
            Number of layers. Default: 2.
        gru_kernel_size: int
            Size of the GRU kernel. Default: 1.
        orthogonal_initialization: bool
            Orthogonal initialization is used if set to True. Default: True.
        instance_norm: bool
            Instance norm is used if set to True. Default: False.
        dense_connect: int
            Number of dense connections.
        replication_padding: bool
            If set to true replication padding is applied.
        norm_groups: int,
            Number of normalization groups.
        """
        super().__init__()
        self.convgru = Conv2dGRU(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            gru_kernel_size=gru_kernel_size,
            orthogonal_initialization=orthogonal_initialization,
            instance_norm=instance_norm,
            dense_connect=dense_connect,
            replication_padding=replication_padding,
        )
        self.norm_groups = norm_groups

    @staticmethod
    def norm(input_data: torch.Tensor, num_groups: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, num_groups, -1)

        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)

        output = (input_data - mean) / std
        output = output.reshape(b, c, h, w)

        return output, mean, std

    @staticmethod
    def unnorm(input_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, num_groups: int) -> torch.Tensor:
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, num_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    def forward(
        self,
        cell_input: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes :class:`NormConv2dGRU` forward pass given tensors `cell_input` and `previous_state`.
        It performs group normalization on the input before the forward pass to the Conv2dGRU.
        Output of Conv2dGRU is then un-normalized.
        Parameters
        ----------
        cell_input: torch.Tensor
            Input tensor.
        previous_state: torch.Tensor
            Tensor of previous hidden state.
        Returns
        -------
        out, new_states: (torch.Tensor, torch.Tensor)
            Output and new states.
        """
        # Normalize
        cell_input, mean, std = self.norm(cell_input, self.norm_groups)
        # Pass normalized input
        cell_input, previous_state = self.convgru(cell_input, previous_state)
        # Unnormalize output
        cell_input = self.unnorm(cell_input, mean, std, self.norm_groups)

        return cell_input, previous_state


#===============================================
# basic module
#===============================================


class RecurrentInit(nn.Module):
    """Recurrent State Initializer (RSI) module of Recurrent Variational Network as presented in [1]_.
    The RSI module learns to initialize the recurrent hidden state :math:`h_0`, input of the first RecurrentVarNetBlock of the RecurrentVarNet.
    References
    ----------
    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org, http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...],
        dilations: Tuple[int, ...],
        depth: int = 2,
        multiscale_depth: int = 1,
    ):
        """Inits :class:`RecurrentInit`.
        Parameters
        ----------
        in_channels: int
            Input channels.
        out_channels: int
            Number of hidden channels of the recurrent unit of RecurrentVarNet Block.
        channels: tuple
            Channels :math:`n_d` in the convolutional layers of initializer.
        dilations: tuple
            Dilations :math:`p` of the convolutional layers of the initializer.
        depth: int
            RecurrentVarNet Block number of layers :math:`n_l`.
        multiscale_depth: 1
            Number of feature layers to aggregate for the output, if 1, multi-scale context aggregation is disabled.
        """
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = in_channels
        for (curr_channels, curr_dilations) in zip(channels, dilations):
            block = [
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            ]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for _ in range(depth):
            block = [nn.Conv2d(tch, out_channels, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes initialization for recurrent unit given input `x`.
        Parameters
        ----------
        x: torch.Tensor
            Initialization for RecurrentInit.
        Returns
        -------
        out: torch.Tensor
            Initial recurrent hidden state from input `x`.
        """

        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)
        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)
        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        out = torch.stack(output_list, dim=-1)
        return out


#=====================================================
# main network
#=====================================================
class RecurrentVarNet(nn.Module):
    """Recurrent Variational Network implementation as presented in [1]_.
    References
    ----------
    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org, http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_steps: int = 4,
        recurrent_hidden_channels: int = 96,
        recurrent_num_layers: int = 4,
        no_parameter_sharing: bool = True,
        learned_initializer: bool = True,
        initializer_initialization: Optional[str] = 'sense',
        initializer_channels: Optional[Tuple[int, ...]] = (32, 32, 64, 64),
        initializer_dilations: Optional[Tuple[int, ...]] = (1, 1, 2, 4),
        initializer_multiscale: int = 3,
        normalized: bool = False,
        shift:bool = False,
        **kwargs,
    ):
        """Inits :class:`RecurrentVarNet`.
        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_steps: int
            Number of iterations :math:`T`.
        in_channels: int
            Input channel number. Default is 2 for complex data.
        recurrent_hidden_channels: int
            Hidden channels number for the recurrent unit of the RecurrentVarNet Blocks. Default: 64.
        recurrent_num_layers: int
            Number of layers for the recurrent unit of the RecurrentVarNet Block (:math:`n_l`). Default: 4.
        no_parameter_sharing: bool
            If False, the same :class:`RecurrentVarNetBlock` is used for all num_steps. Default: True.
        learned_initializer: bool
            If True an RSI module is used. Default: False.
        initializer_initialization: str, Optional
            Type of initialization for the RSI module. Can be either 'sense', 'zero-filled' or 'input-image'.
            Default: None.
        initializer_channels: tuple
            Channels :math:`n_d` in the convolutional layers of the RSI module. Default: (32, 32, 64, 64).
        initializer_dilations: tuple
            Dilations :math:`p` of the convolutional layers of the RSI module. Default: (1, 1, 2, 4).
        initializer_multiscale: int
            RSI module number of feature layers to aggregate for the output, if 1, multi-scale context aggregation
            is disabled. Default: 1.
        normalized: bool
            If True, :class:`NormConv2dGRU` will be used as a regularizer in the :class:`RecurrentVarNetBlocks`. Default: False.
        """
        super(RecurrentVarNet, self).__init__()

        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        self.initializer: Optional[nn.Module] = None
        if (
            learned_initializer
            and initializer_initialization is not None
            and initializer_channels is not None
            and initializer_dilations is not None
        ):
            if initializer_initialization not in [
                "sense",
                "input_image",
                "zero_filled",
            ]:
                raise ValueError(
                    f"Unknown initializer_initialization. Expected `sense`, `'input_image` or `zero_filled`."
                    f"Got {initializer_initialization}."
                )
            self.initializer_initialization = initializer_initialization
            self.initializer = RecurrentInit(
                in_channels,
                recurrent_hidden_channels,
                channels=initializer_channels,
                dilations=initializer_dilations,
                depth=recurrent_num_layers,
                multiscale_depth=initializer_multiscale,
            )
        self.num_steps = num_steps
        self.no_parameter_sharing = no_parameter_sharing
        self.block_list = nn.ModuleList()

        for _ in range(self.num_steps if self.no_parameter_sharing else 1):
            self.block_list.append(
                RecurrentVarNetBlock(
                    in_channels=in_channels,
                    hidden_channels=recurrent_hidden_channels,
                    num_layers=recurrent_num_layers,
                    normalized=normalized,
                    shift = shift
                )
            )

        self.sens_net = SensitivityModel(chans = 8, num_pools = 4, shift=shift)

        # shape (B, coils, H, W, 2)
        self._coil_dim = 1
        self._spatial_dims = (2, 3)

        self.shift = shift

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        r"""Computes sense initialization :math:`x_{\text{SENSE}}`:
        .. math::
            x_{\text{SENSE}} = \sum_{k=1}^{n_c} {S^{k}}^* \times y^k
        where :math:`y^k` denotes the data from coil :math:`k`.
        Parameters
        ----------
        kspace: torch.Tensor
            k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        Returns
        -------
        input_image: torch.Tensor
            Sense initialization :math:`x_{\text{SENSE}}`.
        """
        input_image = T.complex_mul(
            T.conjugate(sensitivity_map),
            T.ifft2(kspace, shift=self.shift),
        )
        input_image = input_image.sum(self._coil_dim)

        return input_image

    def forward(
        self,
        zf: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`RecurrentVarNet`.
        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        Returns
        -------
        kspace_prediction: torch.Tensor
            k-space prediction.
        """

        previous_state: Optional[torch.Tensor] = None

        sensitivity_map = self.sens_net(masked_kspace, sampling_mask)

        if self.initializer is not None:
            if self.initializer_initialization == "sense":
                initializer_input_image = self.compute_sense_init(
                    kspace=masked_kspace,
                    sensitivity_map=sensitivity_map,
                ).unsqueeze(self._coil_dim)
            elif self.initializer_initialization == "input_image":
                if "initial_image" not in kwargs:
                    raise ValueError(
                        f"`'initial_image` is required as input if initializer_initialization "
                        f"is {self.initializer_initialization}."
                    )
                initializer_input_image = kwargs["initial_image"].unsqueeze(self._coil_dim)
            elif self.initializer_initialization == "zero_filled":
                initializer_input_image = T.ifft2(masked_kspace, shift=self.shift)

            previous_state = self.initializer(
                T.fft2(initializer_input_image, shift=self.shift)
                .sum(self._coil_dim)
                .permute(0, 3, 1, 2)
            )

        kspace_prediction = masked_kspace.clone()
        for step in range(self.num_steps):
            block = self.block_list[step] if self.no_parameter_sharing else self.block_list[0]
            kspace_prediction, previous_state = block(
                kspace_prediction,
                masked_kspace,
                sampling_mask,
                sensitivity_map,
                previous_state,
                self._coil_dim,
                self._spatial_dims,
            )

        output = T.ifft2(kspace_prediction, shift=self.shift) #(B, coils, H, W, 2)
        #output = output.reshape(masked_kspace.shape[0], -1, masked_kspace.shape[2], masked_kspace.shape[3]) #(B, coils*2, H, W)

        return output


class RecurrentVarNetBlock(nn.Module):
    r"""Recurrent Variational Network Block :math:`\mathcal{H}_{\theta_{t}}` as presented in [1]_.
    References
    ----------
    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org, http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 4,
        normalized: bool = False,
        shift: bool = False,
    ):
        """Inits RecurrentVarNetBlock.
        Parameters
        ----------
        forward_operator: Callable
            Forward Fourier Transform.
        backward_operator: Callable
            Backward Fourier Transform.
        in_channels: int,
            Input channel number. Default is 2 for complex data.
        hidden_channels: int,
            Hidden channels. Default: 64.
        num_layers: int,
            Number of layers of :math:`n_l` recurrent unit. Default: 4.
        normalized: bool
            If True, :class:`NormConv2dGRU` will be used as a regularizer. Default: False.
        """
        super().__init__()

        self.learning_rate = nn.Parameter(torch.tensor([1.0]))  # :math:`\alpha_t`
        regularizer_params = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "replication_padding": True,
        }
        # Recurrent Unit of RecurrentVarNet Block :math:`\mathcal{H}_{\theta_t}`
        self.regularizer = (
            NormConv2dGRU(**regularizer_params) if normalized else Conv2dGRU(**regularizer_params)  # type: ignore
        )
        self.shift = shift



    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        hidden_state: Union[None, torch.Tensor],
        coil_dim: int = 1,
        spatial_dims: Tuple[int, int] = (2, 3),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward pass of RecurrentVarNetBlock.
        Parameters
        ----------
        current_kspace: torch.Tensor
            Current k-space prediction of shape (N, coil, height, width, complex=2).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor or None
            Recurrent unit hidden state of shape (N, hidden_channels, height, width, num_layers) if not None. Optional.
        coil_dim: int
            Coil dimension. Default: 1.
        spatial_dims: tuple of ints
            Spatial dimensions. Default: (2, 3).
        Returns
        -------
        new_kspace: torch.Tensor
            New k-space prediction of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor
            Next hidden state of shape (N, hidden_channels, height, width, num_layers).
        """

        kspace_error = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_kspace - masked_kspace,
        )

        recurrent_term = T.reduce_operator(
            T.ifft2(current_kspace, shift=self.shift),
            sensitivity_map,
            dim=coil_dim,
        ).permute(0, 3, 1, 2)

        recurrent_term, hidden_state = self.regularizer(recurrent_term, hidden_state)  # :math:`w_t`, :math:`h_{t+1}`
        recurrent_term = recurrent_term.permute(0, 2, 3, 1)

        recurrent_term = T.fft2(
            T.expand_operator(recurrent_term, sensitivity_map, dim=coil_dim),
            shift=self.shift,
        )

        new_kspace = current_kspace - self.learning_rate * kspace_error + recurrent_term

        return new_kspace, hidden_state  # type: ignore
