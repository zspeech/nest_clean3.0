# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is adapted from NeMo's conformer_encoder.py, conformer_modules.py,
# multi_head_attention.py, and subsampling.py for the nest_ssl_project.
# All NeMo dependencies have been removed.

import math
import random
import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

__all__ = ['ConformerEncoder']

INF_VAL = 10000.0


def avoid_float16_autocast_context():
    """
    If the current autocast context is float16, cast it to bfloat16
    if available (unless we're in jit) or float32.
    This matches NeMo's implementation in nemo.utils.cast_utils.
    """
    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return torch.amp.autocast('cuda', dtype=torch.float32)

        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast('cuda', dtype=torch.bfloat16)
        else:
            return torch.amp.autocast('cuda', dtype=torch.float32)
    else:
        return nullcontext()


# ============================================================================
# Stub classes to replace NeMo dependencies
# ============================================================================

class NeuralModule(nn.Module):
    """Base class replacing NeMo's NeuralModule."""
    pass


class Exportable:
    """Stub class replacing NeMo's Exportable mixin."""
    pass


class AccessMixin:
    """Stub class replacing NeMo's AccessMixin with basic interctc support."""
    
    def __init__(self):
        self._access_cfg = {}
        self._registry = {}
        self._access_enabled = False
    
    def is_access_enabled(self, guid=None):
        return getattr(self, '_access_enabled', False)
    
    def set_access_enabled(self, access_enabled, guid=None):
        self._access_enabled = access_enabled
    
    def update_access_cfg(self, cfg, guid=None):
        if not hasattr(self, '_access_cfg'):
            self._access_cfg = {}
        self._access_cfg.update(cfg)
    
    @property
    def access_cfg(self):
        return getattr(self, '_access_cfg', {})
    
    def register_accessible_tensor(self, name, tensor):
        if not hasattr(self, '_registry'):
            self._registry = {}
        if name not in self._registry:
            self._registry[name] = []
        self._registry[name].append(tensor)
    
    def get_module_registry(self, module):
        return {id(module): getattr(module, '_registry', {})}
    
    def reset_registry(self):
        self._registry = {}


@dataclass
class CacheAwareStreamingConfig:
    """Configuration for cache-aware streaming inference."""
    chunk_size: int = 0  # the size of each chunk at each step
    shift_size: int = 0  # the size of the shift in each step
    cache_drop_size: int = 0  # the number of steps to drop from the cache
    last_channel_cache_size: int = 0  # the size of the needed cache for last channel layers
    valid_out_len: int = 0  # the number of valid output steps
    pre_encode_cache_size: int = 0  # the size of the needed cache for pre-encoding
    drop_extra_pre_encoded: int = 0  # the number of steps to get dropped after the pre-encoding layer
    last_channel_num: int = 0  # number of the last channel layers which need caching
    last_time_num: int = 0  # number of the last time layers which need caching


class StreamingEncoder(ABC):
    """Abstract base class for streaming encoders."""
    
    @abstractmethod
    def setup_streaming_params(self, max_look_ahead: int = 10000):
        """Set up streaming parameters."""
        pass

    @abstractmethod
    def get_initial_cache_state(self, batch_size, dtype, device, max_dim):
        """Get initial cache state for streaming."""
        pass

    @staticmethod
    def to_numpy(tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def cache_aware_stream_step(
        self,
        processed_signal,
        processed_signal_length=None,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        keep_all_outputs=True,
        drop_extra_pre_encoded=None,
        bypass_pre_encode=False,
    ):
        """Perform a single streaming step with caching."""
        if self.streaming_cfg is None:
            self.setup_streaming_params()
        if drop_extra_pre_encoded is not None:
            prev_drop_extra_pre_encoded = self.streaming_cfg.drop_extra_pre_encoded
            self.streaming_cfg.drop_extra_pre_encoded = drop_extra_pre_encoded
        else:
            prev_drop_extra_pre_encoded = None

        if processed_signal_length is None:
            processed_signal_length = processed_signal.new_full(processed_signal.size(0), processed_signal.size(-1))

        encoder_output = self(
            audio_signal=processed_signal,
            length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
        )

        encoder_output = self.streaming_post_process(encoder_output, keep_all_outputs=keep_all_outputs)

        if prev_drop_extra_pre_encoded is not None:
            self.streaming_cfg.drop_extra_pre_encoded = prev_drop_extra_pre_encoded

        return encoder_output


def typecheck():
    """Stub decorator replacing NeMo's typecheck."""
    def decorator(func):
        return func
    return decorator


# ============================================================================
# Utility functions
# ============================================================================

def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


def apply_channel_mask(tensor, mask):
    """Apply mask to tensor with channel dimension.
    
    Note: Cannot use inplace operation (mul_) here because the tensor may be needed
    for gradient computation in backward pass. Using regular multiplication with
    broadcasting instead.
    """
    # tensor: (batch, channels, time, features)
    # mask: (batch, time, features)
    # Use broadcasting: mask.unsqueeze(1) creates (batch, 1, time, features)
    # which automatically broadcasts to (batch, channels, time, features)
    return tensor * mask.unsqueeze(1)


def calculate_conv_output_size(input_size: torch.Tensor, kernel_size: int, stride: int, padding: tuple):
    """Calculate exact output size after convolution."""
    return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


def compute_stochastic_depth_drop_probs(
    num_layers: int,
    stochastic_depth_drop_prob: float = 0.0,
    stochastic_depth_mode: str = "linear",
    stochastic_depth_start_layer: int = 1,
) -> List[float]:
    """Computes drop probabilities for stochastic depth regularization technique.
    
    Args:
        num_layers (int): number of layers in the network.
        stochastic_depth_drop_prob (float): if non-zero, will randomly drop
            layers during training. The higher this value, the more often layers
            are dropped. Defaults to 0.0.
        stochastic_depth_mode (str): can be either "linear" or "uniform". If
            set to "uniform", all layers have the same probability of drop. If
            set to "linear", the drop probability grows linearly from 0 for the
            first layer to the desired value for the final layer. Defaults to
            "linear".
        stochastic_depth_start_layer (int): starting layer for stochastic depth.
            All layers before this will never be dropped. Note that drop
            probability will be adjusted accordingly if mode is "linear" when
            start layer is > 1. Defaults to 1.
    Returns:
        List[float]: list of drop probabilities for all layers
    """
    if not (0 <= stochastic_depth_drop_prob < 1.0):
        raise ValueError("stochastic_depth_drop_prob has to be in [0, 1).")
    if not (1 <= stochastic_depth_start_layer <= num_layers):
        raise ValueError("stochastic_depth_start_layer has to be in [1, num layers].")

    # Layers before `stochastic_depth_start_layer` are never dropped
    layer_drop_probs = [0.0] * stochastic_depth_start_layer

    # Layers starting with `stochastic_depth_start_layer` may be dropped
    if (L := num_layers - stochastic_depth_start_layer) > 0:
        if stochastic_depth_mode == "linear":
            layer_drop_probs += [l / L * stochastic_depth_drop_prob for l in range(1, L + 1)]
        elif stochastic_depth_mode == "uniform":
            layer_drop_probs += [stochastic_depth_drop_prob] * L
        else:
            raise ValueError(
                f'stochastic_depth_mode has to be one of ["linear", "uniform"]. Current value: {stochastic_depth_mode}'
            )
    return layer_drop_probs


# ============================================================================
# Causal Convolutions (copied from NeMo)
# ============================================================================

class CausalConv2D(nn.Conv2d):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be set as None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1

        padding = 0
        super(CausalConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, x):
        x = F.pad(x, pad=(self._left_padding, self._right_padding, self._left_padding, self._right_padding))
        x = super().forward(x)
        return x


class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to locations on its right or left
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super(CausalConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1) :]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache


# ============================================================================
# Subsampling Modules (copied from NeMo)
# ============================================================================

class MaskedConvSequential(nn.Sequential):
    """Sequential module that applies convolutions with masking and length updates."""
    
    def forward(self, x, lengths):
        # Convert input (batch, time, features) to conv format
        x = x.unsqueeze(1)  # (batch, 1, time, features)
        current_lengths = lengths.clone().float()
        mask = self._create_mask(x, current_lengths.long())

        # Process through each layer with mask propagation
        for i, layer in enumerate(self):
            # Apply current mask before layer
            x = apply_channel_mask(x, mask)

            # Apply layer
            x = layer(x)

            # Update lengths for stride operations with proper padding
            if hasattr(layer, 'stride') and layer.stride != (1, 1):
                if hasattr(layer, "_left_padding"):
                    padding = (layer._left_padding, layer._right_padding)  # CausalConv2D
                else:
                    padding = layer.padding
                current_lengths = calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], padding
                )
                mask = self._create_mask(x, current_lengths.long())

        # Final masking
        x = apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _create_mask(self, tensor, lengths):
        """Create mask matching tensor dimensions.
        
        Optimized: avoid multiple expand calls by using broadcasting.
        """
        batch_size, channels, time, features = tensor.shape
        # Create time mask: (batch, time)
        time_mask = torch.arange(time, device=tensor.device, dtype=torch.long).expand(batch_size, time) < lengths.unsqueeze(1)
        # Expand to (batch, time, features) using broadcasting
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)


class ConvSubsampling(torch.nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach.
    
    Copied directly from NeMo's ConvSubsampling implementation.
    """

    def __init__(
        self,
        subsampling,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        super(ConvSubsampling, self).__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        if subsampling == 'vggnet':
            self._stride = 2
            self._kernel_size = 2
            self._ceil_mode = True

            self._left_padding = 0
            self._right_padding = 0

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.MaxPool2d(
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        ceil_mode=self._ceil_mode,
                    )
                )
                in_channels = conv_channels

        elif subsampling == 'dw_striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    )
                )
            else:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        )
                    )

                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        if subsampling in ["vggnet", "dw_striding", "striding"]:
            in_length = torch.tensor(feat_in, dtype=torch.float)
            out_length = calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
            self.conv2d_subsampling = True
        else:
            self.out = None
            self.conv2d_subsampling = False

        self.conv = MaskedConvSequential(*layers)

    def forward(self, x, lengths):
        out_lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        # Transpose to Channel First mode
        if not self.conv2d_subsampling:
            x = x.transpose(1, 2)

        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
                if torch.numel(x) > x_ceil:
                    need_to_split = True
                else:
                    need_to_split = False
            else:
                need_to_split = True

            if need_to_split:
                x, lengths, success = self.conv_split_by_batch(x, lengths)
                if not success:  # if unable to split by batch, try by channel
                    if self._subsampling == 'dw_striding':
                        # TODO: implement lengths inside conv_split_by_channel
                        x = self.conv_split_by_channel(x)
                        lengths = out_lengths
                    else:
                        x, lengths = self.conv(x, lengths)  # try anyway
            else:
                x, lengths = self.conv(x, lengths)
        else:
            x, lengths = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)

        return x, lengths

    def conv_split_by_batch(self, x, lengths):
        """Tries to split input by batch, run conv and concat results"""
        b, *_ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, lengths, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, lengths, False

        ans = [
            self.conv(chunk, ln)
            for chunk, ln in zip(
                torch.split(x, new_batch_size, 0),
                torch.split(lengths, new_batch_size, 0),
            )
        ]
        return torch.cat([a[0] for a in ans]), torch.cat([a[1] for a in ans]), True

    def conv_split_by_channel(self, x):
        """For dw convs, tries to split input by time, run conv and concat results"""

        # Note: this method doesn't use the convolution masking implemented in MaskedConvSequential
        x = x.unsqueeze(0)
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(torch.numel(x) / 2**31, 2))
                cf = 2**p

            new_c = int(c // cf)
            if new_c == 0:
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                new_t = 1

            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c, x)  # conv2D, depthwise

            # splitting pointwise convs by time
            x = torch.cat([self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_t, 2)], 2)  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk, pad=(self._kernel_size - 1, self._stride - 1, self._kernel_size - 1, self._stride - 1)
                )
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)


class StackingSubsampling(torch.nn.Module):
    """Stacking subsampling which simply stacks consecutive frames to reduce the sampling rate"""

    def __init__(self, subsampling_factor, feat_in, feat_out, norm=False):
        super(StackingSubsampling, self).__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out)
        if norm:
            self.pre_norm = LayerNorm(feat_in)
        else:
            self.pre_norm = None

    def forward(self, x, lengths):
        b, t, h = x.size()
        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = self.proj_out(x)
        lengths = torch.div(lengths + pad_size, self.subsampling_factor, rounding_mode='floor')
        return x, lengths


class SubsamplingReductionModule(nn.Module):
    """Downsamples the audio signal in time dimension after some encoder layers."""

    def __init__(self, reduction: str, d_model: int, reduction_factor: int = 2):
        super().__init__()

        assert reduction in ['pooling', 'striding']

        self.reduction = reduction
        self.d_model = d_model
        self._sampling_num = int(math.log(reduction_factor, 2))

        if reduction == 'pooling':
            self.reduction_enc = nn.MaxPool1d(kernel_size=reduction_factor)
            self.padding = 0
            self.kernel_size = self.reduction_enc.kernel_size
            self.stride = self.reduction_enc.stride
        elif reduction == 'striding':
            self.reduction_enc = ConvSubsampling(
                subsampling='striding',
                subsampling_factor=reduction_factor,
                feat_in=d_model,
                feat_out=d_model,
                conv_channels=d_model,
                activation=nn.ReLU(),
                is_causal=False,
            )

    def forward(self, x, lengths):
        """Shapes:
        - x: [B, T, C]
        - lengths: [B]
        """
        if self.reduction == 'striding':
            x, lengths = self.reduction_enc(x=x, lengths=lengths)
        else:
            x = torch.transpose(x, 1, 2)  # [B, C, T]
            lengths = calc_length(
                lengths=lengths,
                all_paddings=self.padding,
                kernel_size=self.kernel_size,
                stride=self.stride,
                ceil_mode=False,
                repeat_num=self._sampling_num,
            )
            x = self.reduction_enc(x)
            x = torch.transpose(x, 1, 2)  # [B, T, C]

        return x, lengths


# ============================================================================
# Positional Encoding (copied from NeMo)
# ============================================================================

class PositionalEncoding(torch.nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None

    def create_pe(self, positions, dtype):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(INF_VAL) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        if hasattr(self, 'pe') and self.pe.size(1) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x: torch.Tensor, cache_len=0):
        """Adds positional encoding."""
        input_len = x.size(1) + cache_len
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, :input_len]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        x = x + pos_emb
        return self.dropout(x), pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for TransformerXL's layers."""

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
        # positions would be from negative numbers to positive
        # positive positions would be used for left positions and negative for right positions
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        """Compute positional encoding."""
        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


class LocalAttRelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for sliding window attention or chunked attention.
    
    Args:
        att_context_size (List[int]): List of 2 ints corresponding to left and right attention context sizes.
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(self, att_context_size, **kwargs):
        super(LocalAttRelPositionalEncoding, self).__init__(**kwargs)
        self.left_context = att_context_size[0]
        self.right_context = att_context_size[1]

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings only at the beginning"""
        if hasattr(self, 'pe'):
            return

        positions = torch.arange(
            self.left_context, -self.right_context - 1, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        """Compute positional encoding."""
        if self.xscale:
            x = x * self.xscale

        end_pos = self.left_context + self.right_context + 1
        pos_emb = self.pe[:, :end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


# ============================================================================
# Multi-Head Attention (copied from NeMo)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer of Transformer."""

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        max_cache_len=0,
        use_bias=True,
    ):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadAttention, self).__init__()
        self.cache_drop_size = None
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._max_cache_len = max_cache_len

    def forward_qkv(self, query, key, value):
        """Transforms query, key and value."""
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector."""
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -INF_VAL)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)

    def forward(self, query, key, value, mask, pos_emb=None, cache=None):
        """Compute 'Scaled Dot Product Attention'."""
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
            out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache

    def update_cache(self, key, value, query, cache):
        if cache is not None:
            key = value = torch.cat([cache, key], dim=1)
            q_keep_size = query.shape[1] - self.cache_drop_size
            cache = torch.cat([cache[:, q_keep_size:, :], query[:, :q_keep_size, :]], dim=1)
        return key, value, query, cache


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding."""

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        pos_bias_u,
        pos_bias_v,
        max_cache_len=0,
        use_bias=True,
    ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            max_cache_len=max_cache_len,
            use_bias=use_bias,
        )
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable biases are used in matrix c and matrix d
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding."""
        b, h, qlen, pos_len = x.size()
        x = torch.nn.functional.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)
        return x

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding."""
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)

            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)

            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            # compute matrix b and matrix d
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)

            # drops extra elements in the matrix_bd to match the matrix_ac's size
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
            scores = (matrix_ac + matrix_bd) / self.s_d_k

            out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache


class RelPositionMultiHeadAttentionLongformer(RelPositionMultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with sliding window local+global attention from Longformer.
    
    Partially adapted from allenai (https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py)
    and huggingface (https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py)
    
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        pos_bias_u (Tensor): the positional bias matrix U
        pos_bias_v (Tensor): the positional bias matrix V
        att_context_size (List[int]): List of 2 ints corresponding to left and right attention context sizes.
        max_cache_len (int): the maximum size of cache
        global_tokens (int): number of tokens to be used for global attention
        global_tokens_spacing (int): how far apart the global tokens are
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate
        use_bias (bool): whether to apply bias in linear and conv layers of MultiHeadAttention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        pos_bias_u,
        pos_bias_v,
        att_context_size,
        max_cache_len=0,
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        use_bias=True,
    ):
        """Construct an RelPositionMultiHeadAttentionLongformer object."""
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            max_cache_len=max_cache_len,
            use_bias=use_bias,
        )

        self.att_context_size = att_context_size
        self.global_tokens = global_tokens
        self.global_tokens_spacing = global_tokens_spacing
        self.global_attn_separate = global_attn_separate

        if self.global_attn_separate:
            self.global_q = nn.Linear(n_feat, n_feat, bias=use_bias)
            self.global_k = nn.Linear(n_feat, n_feat, bias=use_bias)
            self.global_v = nn.Linear(n_feat, n_feat, bias=use_bias)

    def forward(self, query, key, value, pad_mask, pos_emb, cache=None):
        """Compute Scaled Dot Product Local Attention with rel. positional encoding using overlapping chunks.
        
        Args:
            query (torch.Tensor): (batch, time, size)
            key (torch.Tensor): (batch, time, size)
            value(torch.Tensor): (batch, time, size)
            pad_mask (torch.Tensor): (batch, time)
            pos_emb (torch.Tensor) : (batch, 2w + 1, size)
            cache (torch.Tensor) : (batch, time_cache, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            n_batch, _, T, _ = q.size()

            w = max(self.att_context_size[0], self.att_context_size[1])
            if w <= 0:
                raise ValueError("When using local attention, context size must be set > 0")
            pad_len = (2 * w - T % (2 * w)) % (2 * w)
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            mask = F.pad(pad_mask, (0, pad_len), value=1.0)

            q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)
            q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)

            diagonal_matrix_ac = self.sliding_chunks_matmul_qk(q_with_bias_u, k, w, padding_value=0.0)

            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)
            diagonal_matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

            start_pos = w - self.att_context_size[0]
            end_pos = w + self.att_context_size[1]

            diagonal_matrix_ac[:, :, :, : self.att_context_size[0]] += diagonal_matrix_bd[
                :, :, :, : self.att_context_size[0]
            ]
            diagonal_matrix_ac[:, :, :, -(self.att_context_size[1] + 1) :] += diagonal_matrix_bd[
                :, :, :, self.att_context_size[0] :
            ]
            scores = diagonal_matrix_ac / self.s_d_k

            scores[:, :, :, :start_pos] = -INF_VAL
            scores[:, :, :, end_pos + 1 :] = -INF_VAL

            mask = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            float_mask = mask.type_as(scores).masked_fill(mask, -INF_VAL)
            ones = float_mask.new_ones(size=float_mask.size())
            d_mask = self.sliding_chunks_matmul_qk(ones, float_mask, w, padding_value=0.0)

            scores += d_mask

            if self.global_tokens > 0:
                if self.global_attn_separate:
                    global_q = self.global_q(query).view(n_batch, -1, self.h, self.d_k)
                    global_k = self.global_k(key).view(n_batch, -1, self.h, self.d_k)
                    global_v = self.global_v(value).view(n_batch, -1, self.h, self.d_k)
                    global_q = global_q.transpose(1, 2)
                    global_k = global_k.transpose(1, 2)
                    global_v = global_v.transpose(1, 2)
                    global_q = F.pad(global_q, (0, 0, 0, pad_len))
                    global_k = F.pad(global_k, (0, 0, 0, pad_len))
                    global_v = F.pad(global_v, (0, 0, 0, pad_len))
                else:
                    global_q, global_k, global_v = q, k, v

                global_q /= self.s_d_k

                is_index_global_attn = torch.zeros_like(pad_mask)
                is_index_global_attn[
                    :, : self.global_tokens * self.global_tokens_spacing : self.global_tokens_spacing
                ] = 1.0

                (
                    max_num_global_attn_indices,
                    is_index_global_attn_nonzero,
                    is_local_index_global_attn_nonzero,
                    is_local_index_no_global_attn_nonzero,
                ) = self._get_global_attn_indices(is_index_global_attn=is_index_global_attn)

                global_key_attn = self._compute_global_key_attn(
                    query=global_q.transpose(1, 2),
                    key=global_k.transpose(1, 2),
                    max_num_global_attn_indices=max_num_global_attn_indices,
                    is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                    is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                ).transpose(1, 2)

                scores = torch.cat((global_key_attn, scores), dim=-1)
                del global_key_attn

            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
            p_attn = self.dropout(attn)

            if self.global_tokens > 0:
                out = self._compute_attn_output_with_global_indices(
                    value=v,
                    attn_probs=p_attn,
                    max_num_global_attn_indices=max_num_global_attn_indices,
                    is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                    is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    w=w,
                )
            else:
                out = self.sliding_chunks_matmul_pv(p_attn, v, w)

            out = out.reshape(n_batch, -1, self.h * self.d_k)[:, :T]

            if self.global_tokens > 0:
                out_global_to_all = self._compute_out_global_to_all(
                    query=global_q,
                    key=global_k,
                    value=global_v,
                    max_num_global_attn_indices=max_num_global_attn_indices,
                    is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                    is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                    is_index_masked=mask,
                )
                out[is_index_global_attn_nonzero] = out_global_to_all

        ret = self.linear_out(out)

        if cache is None:
            return ret
        else:
            return ret, cache

    def _get_global_attn_indices(self, is_index_global_attn: torch.Tensor) -> Tuple:
        """Compute global attention indices."""
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)
        max_num_global_attn_indices = num_global_attn_indices.max()
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _compute_global_key_attn(
        self, key, query, max_num_global_attn_indices, is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero,
    ) -> torch.Tensor:
        """Compute the attention probabilities using only global key vectors."""
        batch_size = key.shape[0]
        key_only_global = key.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)
        key_only_global[is_local_index_global_attn_nonzero] = key[is_index_global_attn_nonzero]
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query, key_only_global))
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self, value, attn_probs, max_num_global_attn_indices,
        is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, w,
    ) -> torch.Tensor:
        """Compute the attention output with global indices."""
        batch_size, time = attn_probs.shape[0], attn_probs.shape[2]
        value = value.transpose(1, 2)
        value_vectors_only_global = value.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value[is_index_global_attn_nonzero]
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.clone(), value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()
        attn_output_without_global = self.sliding_chunks_matmul_pv(attn_probs_without_global, value.transpose(1, 2), w)
        return attn_output_only_global + attn_output_without_global

    def _compute_out_global_to_all(
        self, query, key, value, max_num_global_attn_indices,
        is_local_index_global_attn_nonzero, is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero, is_index_masked,
    ) -> torch.Tensor:
        """Compute the attention output of global tokens attending to all."""
        batch_size = key.shape[0]
        seq_len = key.shape[2]

        global_k = key.reshape(batch_size * self.h, -1, self.d_k)
        global_v = value.reshape(batch_size * self.h, -1, self.d_k)

        global_q = query.transpose(1, 2)
        global_q_from_global = global_q.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)
        global_q_from_global[is_local_index_global_attn_nonzero] = global_q[is_index_global_attn_nonzero]
        global_q_from_global = global_q_from_global.transpose(0, 1).reshape(batch_size * self.h, -1, self.d_k)

        global_attn_scores = torch.bmm(global_q_from_global, global_k.transpose(1, 2))
        global_attn_scores = global_attn_scores.view(batch_size, self.h, max_num_global_attn_indices, seq_len)

        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(global_attn_scores.dtype).min
        global_attn_scores = global_attn_scores.transpose(1, 2)

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked.transpose(2, 3),
            torch.finfo(global_attn_scores.dtype).min,
        )

        global_attn_scores = global_attn_scores.view(batch_size * self.h, max_num_global_attn_indices, seq_len)

        if self.training:
            global_attn_probs_float = nn.functional.softmax(global_attn_scores, dim=-1, dtype=torch.float32)
        else:
            global_attn_probs_float = nn.functional.softmax(global_attn_scores, dim=-1)

        global_attn_probs = self.dropout(global_attn_probs_float)
        global_attn_output = torch.bmm(global_attn_probs, global_v)
        global_attn_output = global_attn_output.view(batch_size, self.h, max_num_global_attn_indices, self.d_k)
        global_attn_output = global_attn_output[
            is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
        ]
        global_attn_output = global_attn_output.reshape(global_attn_output.shape[0], -1)
        return global_attn_output

    def _skew(self, x: torch.Tensor, direction: List[int], padding_value: float) -> torch.Tensor:
        """Convert diagonals into columns."""
        x_padded = F.pad(x, direction, value=padding_value)
        x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        return x_padded

    def _skew2(self, x: torch.Tensor, padding_value: float) -> torch.Tensor:
        """Shift every row 1 step to right converting columns into diagonals."""
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1), value=padding_value)
        x = x.view(B, C, -1)
        x = x[:, :, :-M]
        x = x.view(B, C, M, M + L)
        x = x[:, :, :, :-1]
        return x

    def _chunk_overlap(self, x: torch.Tensor, w: int) -> torch.Tensor:
        """Convert into overlapping chunks."""
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @lru_cache()
    def _get_invalid_locations_mask(self, w: int, device: str):
        diagonals_list = []
        for j in range(-w, 1):
            diagonal_mask = torch.zeros(w, device='cpu', dtype=torch.uint8)
            diagonal_mask[:-j] = 1
            diagonals_list.append(diagonal_mask)
        mask = torch.stack(diagonals_list, dim=-1)
        mask = mask[None, None, :, :]
        ending_mask = mask.flip(dims=(2, 3)).bool().to(device)
        return mask.bool().to(device), ending_mask

    def mask_invalid_locations(self, input_tensor: torch.Tensor, w: int):
        """Mask locations invalid for the sliding window attention."""
        beginning_mask, ending_mask = self._get_invalid_locations_mask(w, input_tensor.device)
        seq_len = input_tensor.size(2)
        beginning_input = input_tensor[:, :, :w, : w + 1]
        beginning_mask = beginning_mask[:, :, :seq_len].expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask, -float('inf'))
        ending_input = input_tensor[:, :, -w:, -(w + 1) :]
        ending_mask = ending_mask[:, :, -seq_len:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float('inf'))

    def sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float) -> torch.Tensor:
        """Matrix multiplication of query x key tensors using with a sliding window attention pattern."""
        bsz, num_heads, seqlen, head_dim = q.size()
        assert seqlen % (w * 2) == 0
        assert q.size() == k.size()

        chunks_count = seqlen // w - 1
        q = q.reshape(bsz * num_heads, seqlen, head_dim)
        k = k.reshape(bsz * num_heads, seqlen, head_dim)

        chunk_q = self._chunk_overlap(q, w)
        chunk_k = self._chunk_overlap(k, w)

        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)

        diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, : w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, : w + 1]
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1) : -1, w + 1 :]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, : w - 1, 1 - w :]

        diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1)
        self.mask_invalid_locations(diagonal_attn, w)
        return diagonal_attn

    def sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int):
        """Same as sliding_chunks_matmul_qk but for prob and value tensors."""
        bsz, num_heads, seqlen, head_dim = v.size()
        chunks_count = seqlen // w - 1
        chunk_prob = prob.reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)

        v = v.reshape(bsz * num_heads, seqlen, head_dim)
        padded_v = F.pad(v, (0, 0, w, w), value=-1)

        chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

        skewed_prob = self._skew2(chunk_prob, padding_value=0)
        context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
        return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)


# ============================================================================
# Conformer Modules (copied from NeMo)
# ============================================================================

class Swish(nn.SiLU):
    """
    Swish activation function introduced in 'https://arxiv.org/abs/1710.05941'
    Mathematically identical to SiLU. See note in nn.SiLU for references.
    """
    pass


class ConformerFeedForward(nn.Module):
    """Feed-forward module of Conformer model."""

    def __init__(self, d_model, d_ff, dropout, activation=Swish(), use_bias=True):
        super(ConformerFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = nn.Linear(d_model, d_ff, bias=self.use_bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=self.use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model."""

    def __init__(
        self,
        d_model,
        kernel_size,
        norm_type='batch_norm',
        conv_context_size=None,
        pointwise_activation='glu_',
        use_bias=True,
    ):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.use_bias = use_bias

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        self.pointwise_activation = pointwise_activation
        dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=self.use_bias,
        )

        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        elif norm_type == 'instance_norm':
            self.batch_norm = nn.InstanceNorm1d(dw_conv_input_dim)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim)
        elif norm_type.startswith('group_norm'):
            num_groups = int(norm_type.replace("group_norm", ""))
            self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # Compute the activation function or use GLU for original Conformer
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache


class ConformerLayer(torch.nn.Module, AccessMixin):
    """A single block of the Conformer encoder."""

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=None,
        use_bias=True,
    ):
        super(ConformerLayer, self).__init__()

        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        if att_context_size is None:
            att_context_size = [-1, -1]

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]

        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            self.self_attn = RelPositionMultiHeadAttentionLongformer(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                att_context_size=att_context_size,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                use_bias=use_bias,
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
            )
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos']"
            )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
            cache_last_channel (torch.tensor) : cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : cache for convolutional layers (B, d_model, T_cache)
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, cache=cache_last_channel)
        else:
            x = None

        if x is not None and cache_last_channel is not None:
            (x, cache_last_channel) = x

        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)

        if cache_last_channel is None:
            return x
        else:
            return x, cache_last_channel, cache_last_time


# ============================================================================
# Conformer Encoder (copied from NeMo)
# ============================================================================

class ConformerEncoder(NeuralModule, StreamingEncoder, Exportable, AccessMixin):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100
    """

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style='regular',
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        sync_max_audio_length: bool = True,
        # Additional params that may come from config but are not used in this simplified version
        **kwargs,
    ):
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor
        self.sync_max_audio_length = sync_max_audio_length

        self.self_attention_model = self_attention_model
        self.global_tokens = global_tokens
        self.global_attn_separate = global_attn_separate
        self.global_tokens_spacing = global_tokens_spacing

        # Setting up the att_context_size
        (
            self.att_context_size_all,
            self.att_context_size,
            self.att_context_probs,
            self.conv_context_size,
        ) = self._calc_context_sizes(
            att_context_style=att_context_style,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            conv_context_size=conv_context_size,
            conv_kernel_size=conv_kernel_size,
        )

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        # Subsampling
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    norm=True if subsampling == 'stacking_norm' else False,
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                    activation=nn.ReLU(True),
                    is_causal=causal_downsampling,
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        # Reduction
        if reduction and reduction_factor > 1:
            assert reduction_position >= -1 and reduction_position < n_layers
            self.reduction_subsampling = SubsamplingReductionModule(
                reduction=reduction,
                d_model=d_model,
                reduction_factor=reduction_factor,
            )
            self.reduction_position = reduction_position
        else:
            self.reduction_subsampling = None
            self.reduction_position = None

        self._feat_out = d_model

        # Biases for relative positional encoding
        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        # Positional encodings
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            if max(self.att_context_size) <= 0:
                raise ValueError("When using local attention, context size must be set > 0")
            self.pos_enc = LocalAttRelPositionalEncoding(
                att_context_size=self.att_context_size,
                d_model=d_model,
                dropout_rate=dropout,  # NeMo uses dropout, not dropout_pre_encoder for local attn
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout_pre_encoder, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                conv_context_size=self.conv_context_size,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_context_size=self.att_context_size,
                use_bias=use_bias,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model

        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

        self.setup_streaming_params()
        self.export_cache_support = False

        self.layer_drop_probs = compute_stochastic_depth_drop_probs(
            len(self.layers), stochastic_depth_drop_prob, stochastic_depth_mode, stochastic_depth_start_layer
        )
        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_capture_at_layers = None

    def forward_for_export(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        """Forward function for model export."""
        if cache_last_channel is not None:
            cache_last_channel = cache_last_channel.transpose(0, 1)
            cache_last_time = cache_last_time.transpose(0, 1)

        rets = self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        rets = self.streaming_post_process(rets, keep_all_outputs=False)
        if len(rets) == 2:
            return rets
        elif rets[2] is None and rets[3] is None and rets[4] is None:
            return (rets[0], rets[1])
        else:
            return (
                rets[0],
                rets[1],
                rets[2].transpose(0, 1),
                rets[3].transpose(0, 1),
                rets[4],
            )

    def streaming_post_process(self, rets, keep_all_outputs=True):
        """Post-process the output of the forward function for streaming."""
        if len(rets) == 2:
            return rets[0], rets[1], None, None, None

        (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len) = rets

        if cache_last_channel_next is not None and self.streaming_cfg.last_channel_cache_size >= 0:
            if self.streaming_cfg.last_channel_cache_size > 0:
                cache_last_channel_next = cache_last_channel_next[
                    :, :, -self.streaming_cfg.last_channel_cache_size :, :
                ]

        if self.streaming_cfg.valid_out_len > 0 and (not keep_all_outputs or self.att_context_style == "regular"):
            encoded = encoded[:, :, : self.streaming_cfg.valid_out_len]
            encoded_len = torch.clamp(encoded_len, max=self.streaming_cfg.valid_out_len)

        return (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len)

    @typecheck()
    def forward(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
    ):
        """
        Forward function for the ConformerEncoder.
        
        Args:
            audio_signal: Input tensor. Shape depends on bypass_pre_encode:
                - bypass_pre_encode=False: (batch, feat_in, n_frames)
                - bypass_pre_encode=True: (batch, n_frames, d_model)
            length: Length of each sequence in the batch
            cache_last_channel: Cache for attention layers (streaming)
            cache_last_time: Cache for convolution layers (streaming)
            cache_last_channel_len: Length of cache
            bypass_pre_encode: If True, skip pre-encoding (for pre-computed embeddings)
        """
        if not bypass_pre_encode and audio_signal.shape[-2] != self._feat_in:
            raise ValueError(
                f"If bypass_pre_encode is False, audio_signal should have shape "
                f"(batch, {self._feat_in}, n_frame) but got last dimension {audio_signal.shape[-2]}."
            )
        if bypass_pre_encode and audio_signal.shape[-1] != self.d_model:
            raise ValueError(
                f"If bypass_pre_encode is True, audio_signal should have shape "
                f"(batch, n_frame, {self.d_model}) but got last dimension {audio_signal.shape[-1]}."
            )

        if bypass_pre_encode:
            self.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
        else:
            self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
        )

    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
    ):
        """Internal forward function with full streaming support."""
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        # select a random att_context_size with the distribution specified by att_context_probs during training
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        if not bypass_pre_encode:
            audio_signal = torch.transpose(audio_signal, 1, 2)

            if isinstance(self.pre_encode, nn.Linear):
                audio_signal = self.pre_encode(audio_signal)
            else:
                audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
                length = length.to(torch.int64)
                # streaming: drop extra pre-encoded
                if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                    audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                    length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

            if self.reduction_position is not None and cache_last_channel is not None:
                raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
                cache_last_channel_next.append(cache_last_channel_cur)
                cache_last_time_next.append(cache_last_time_cur)

            # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                if should_drop:
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )

            # saving tensors if required for interctc loss
            if self.is_access_enabled(getattr(self, "model_guid", None)):
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    if self.out_proj is not None:
                        lth_audio_signal = self.out_proj(audio_signal)
                    self.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )
                    self.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction at the end
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return audio_signal, length

    def update_max_seq_length(self, seq_length: int, device):
        """Updates the maximum sequence length for the model."""
        # Find global max audio length across all nodes
        if self.sync_max_audio_length and torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)
            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def set_max_audio_length(self, max_audio_length):
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        if self.self_attention_model != "rel_pos_local_attn":
            att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)

            if self.att_context_style == "regular":
                if att_context_size[0] >= 0:
                    att_mask = att_mask.triu(diagonal=-att_context_size[0])
                if att_context_size[1] >= 0:
                    att_mask = att_mask.tril(diagonal=att_context_size[1])
            elif self.att_context_style == "chunked_limited":
                # When right context is unlimited, just the left side of the masking need to get updated
                if att_context_size[1] == -1:
                    if att_context_size[0] >= 0:
                        att_mask = att_mask.triu(diagonal=-att_context_size[0])
                else:
                    chunk_size = att_context_size[1] + 1
                    if att_context_size[0] >= 0:
                        left_chunks_num = att_context_size[0] // chunk_size
                    else:
                        left_chunks_num = 10000

                    chunk_idx = torch.arange(0, max_audio_length, dtype=torch.int, device=att_mask.device)
                    chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                    diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                    chunked_limited_mask = torch.logical_and(
                        torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
                    )
                    att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))
        else:
            att_mask = None

        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).expand(-1, max_audio_length, -1)
            pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def enable_pad_mask(self, on=True):
        """Enables or disables the pad mask."""
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask

    def set_default_att_context_size(self, att_context_size):
        """Sets the default attention context size."""
        if att_context_size not in self.att_context_size_all:
            logging.warning(
                f"att_context_size={att_context_size} is not among the list of the supported "
                f"look-aheads: {self.att_context_size_all}"
            )
        if att_context_size is not None:
            self.att_context_size = att_context_size
        self.setup_streaming_params()

    def setup_streaming_params(
        self,
        chunk_size: int = None,
        shift_size: int = None,
        left_chunks: int = None,
        att_context_size: list = None,
        max_context: int = 10000,
    ):
        """Set up streaming parameters for cache-aware streaming inference."""
        streaming_cfg = CacheAwareStreamingConfig()

        if att_context_size is None:
            att_context_size = self.att_context_size

        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError("chunk_size needs to be a number larger or equal to one.")
            lookahead_steps = chunk_size - 1
            streaming_cfg.cache_drop_size = chunk_size - shift_size
        elif self.att_context_style == "chunked_limited":
            lookahead_steps = att_context_size[1]
            streaming_cfg.cache_drop_size = 0
        elif self.att_context_style == "regular":
            lookahead_steps = att_context_size[1] * self.n_layers + self.conv_context_size[1] * self.n_layers
            streaming_cfg.cache_drop_size = lookahead_steps
        else:
            streaming_cfg.cache_drop_size = 0
            lookahead_steps = None

        if chunk_size is None:
            streaming_cfg.last_channel_cache_size = att_context_size[0] if att_context_size[0] >= 0 else max_context
        else:
            if left_chunks is None:
                streaming_cfg.last_channel_cache_size = (
                    att_context_size[0] if att_context_size[0] >= 0 else max_context
                )
                logging.warning(
                    f"left_chunks is not set. Setting it to default: {streaming_cfg.last_channel_cache_size}."
                )
            else:
                streaming_cfg.last_channel_cache_size = left_chunks * chunk_size

        if hasattr(self.pre_encode, "get_sampling_frames"):
            sampling_frames = self.pre_encode.get_sampling_frames()
        else:
            sampling_frames = 0

        if isinstance(sampling_frames, list):
            streaming_cfg.chunk_size = [
                sampling_frames[0] + self.subsampling_factor * lookahead_steps,
                sampling_frames[1] + self.subsampling_factor * lookahead_steps,
            ]
        else:
            streaming_cfg.chunk_size = sampling_frames * (1 + lookahead_steps) if lookahead_steps else 0

        if isinstance(sampling_frames, list):
            streaming_cfg.shift_size = [
                sampling_frames[0] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
                sampling_frames[1] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
            ]
        else:
            streaming_cfg.shift_size = sampling_frames * (1 + lookahead_steps - streaming_cfg.cache_drop_size) if lookahead_steps else 0

        if isinstance(streaming_cfg.shift_size, list):
            streaming_cfg.valid_out_len = (
                streaming_cfg.shift_size[1] - sampling_frames[1]
            ) // self.subsampling_factor + 1
        else:
            streaming_cfg.valid_out_len = streaming_cfg.shift_size // self.subsampling_factor if streaming_cfg.shift_size else 0

        if hasattr(self.pre_encode, "get_streaming_cache_size"):
            streaming_cfg.pre_encode_cache_size = self.pre_encode.get_streaming_cache_size()
        else:
            streaming_cfg.pre_encode_cache_size = 0

        if isinstance(streaming_cfg.pre_encode_cache_size, list):
            if streaming_cfg.pre_encode_cache_size[1] >= 1:
                streaming_cfg.drop_extra_pre_encoded = (
                    1 + (streaming_cfg.pre_encode_cache_size[1] - 1) // self.subsampling_factor
                )
            else:
                streaming_cfg.drop_extra_pre_encoded = 0
        else:
            streaming_cfg.drop_extra_pre_encoded = streaming_cfg.pre_encode_cache_size // self.subsampling_factor

        for m in self.layers.modules():
            if hasattr(m, "_max_cache_len"):
                if isinstance(m, MultiHeadAttention):
                    m.cache_drop_size = streaming_cfg.cache_drop_size
                if isinstance(m, CausalConv1D):
                    m.cache_drop_size = streaming_cfg.cache_drop_size

        self.streaming_cfg = streaming_cfg

    def get_initial_cache_state(self, batch_size=1, dtype=torch.float32, device=None, max_dim=0):
        """Get initial cache state for streaming inference."""
        if device is None:
            device = next(self.parameters()).device
        if max_dim > 0:
            create_tensor = torch.randn
        else:
            create_tensor = torch.zeros
        last_time_cache_size = self.conv_context_size[0]
        cache_last_channel = create_tensor(
            (
                len(self.layers),
                batch_size,
                self.streaming_cfg.last_channel_cache_size,
                self.d_model,
            ),
            device=device,
            dtype=dtype,
        )
        cache_last_time = create_tensor(
            (len(self.layers), batch_size, self.d_model, last_time_cache_size),
            device=device,
            dtype=dtype,
        )
        if max_dim > 0:
            cache_last_channel_len = torch.randint(
                0,
                min(max_dim, self.streaming_cfg.last_channel_cache_size),
                (batch_size,),
                device=device,
                dtype=torch.int64,
            )
            for i in range(batch_size):
                cache_last_channel[:, i, cache_last_channel_len[i] :, :] = 0
                if cache_last_channel_len[i] == 0:
                    cache_last_time[:, i, :, :] = 0
        else:
            cache_last_channel_len = torch.zeros(batch_size, device=device, dtype=torch.int64)
        return cache_last_channel, cache_last_time, cache_last_channel_len

    def change_attention_model(
        self,
        self_attention_model: str = None,
        att_context_size: List[int] = None,
        update_config: bool = True,
        device: torch.device = None,
    ):
        """
        Update the self_attention_model which changes the positional encoding and attention layers.
        
        Args:
            self_attention_model: 'rel_pos', 'rel_pos_local_attn', or 'abs_pos'
            att_context_size: List of 2 ints for left and right context sizes
            update_config: Whether to update internal config
            device: Device to move new layers to
        """
        if att_context_size:
            att_context_size = list(att_context_size)
        else:
            att_context_size = self.att_context_size

        if self_attention_model is None:
            self_attention_model = self.self_attention_model

        if self_attention_model == 'rel_pos_local_attn' and max(att_context_size) <= 0:
            raise ValueError("When using local attention, context size must be set > 0")

        if self_attention_model == "rel_pos":
            new_pos_enc = RelPositionalEncoding(
                d_model=self.d_model,
                dropout_rate=0.1,
                max_len=self.pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=0.1,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            new_pos_enc = LocalAttRelPositionalEncoding(
                att_context_size=att_context_size,
                d_model=self.d_model,
                dropout_rate=0.1,
                max_len=self.pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=0.1,
            )
        elif self_attention_model == "abs_pos":
            new_pos_enc = PositionalEncoding(
                d_model=self.d_model,
                dropout_rate=0.1,
                max_len=self.pos_emb_max_len,
                xscale=self.xscale,
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        if device is not None:
            new_pos_enc = new_pos_enc.to(device=device)
        del self.pos_enc
        self.pos_enc = new_pos_enc
        self.self_attention_model = self_attention_model
        self.att_context_size = att_context_size
        self.set_max_audio_length(self.pos_emb_max_len)

        for _, m in self.named_modules():
            if type(m) == ConformerLayer:
                if self_attention_model == 'rel_pos':
                    new_attn = RelPositionMultiHeadAttention(
                        n_head=m.n_heads,
                        n_feat=self.d_model,
                        dropout_rate=0.0,
                        max_cache_len=att_context_size[0],
                        pos_bias_u=None,
                        pos_bias_v=None,
                    )
                elif self_attention_model == 'rel_pos_local_attn':
                    new_attn = RelPositionMultiHeadAttentionLongformer(
                        n_head=m.n_heads,
                        n_feat=self.d_model,
                        dropout_rate=0.0,
                        max_cache_len=att_context_size[0],
                        att_context_size=att_context_size,
                        pos_bias_u=None,
                        pos_bias_v=None,
                        global_tokens=self.global_tokens,
                        global_tokens_spacing=self.global_tokens_spacing,
                        global_attn_separate=self.global_attn_separate,
                    )
                elif self_attention_model == 'abs_pos':
                    new_attn = MultiHeadAttention(
                        n_head=m.n_heads,
                        n_feat=self.d_model,
                        dropout_rate=0.0,
                        max_cache_len=att_context_size[0],
                    )
                else:
                    raise ValueError(
                        f"'{self_attention_model}' is not a valid value for 'self_attention_model'"
                    )
                if device is not None:
                    new_attn = new_attn.to(device=device)
                new_attn.load_state_dict(m.self_attn.state_dict(), strict=False)
                del m.self_attn
                m.self_attn = new_attn
                m.self_attention_model = self_attention_model

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        """
        Update the conv_chunking_factor.
        Default is 1 (auto). Set to -1 (disabled) or a power of 2 if OOM in conv subsampling.
        """
        if not hasattr(self.pre_encode, "change_subsampling_conv_chunking_factor"):
            logging.info("Model pre_encoder doesn't have a change_subsampling_conv_chunking_factor method")
            return

        self.pre_encode.change_subsampling_conv_chunking_factor(
            subsampling_conv_chunking_factor=subsampling_conv_chunking_factor
        )

    def _calc_context_sizes(
        self, att_context_size, att_context_probs, att_context_style, conv_context_size, conv_kernel_size
    ):
        """Calculate attention context sizes and convert to standard format."""
        # convert att_context_size to a standard list of lists
        if att_context_size:
            att_context_size_all = list(att_context_size)
            if isinstance(att_context_size_all[0], int):
                att_context_size_all = [att_context_size_all]
            for i, att_cs in enumerate(att_context_size_all):
                if isinstance(att_cs, (list, tuple)):
                    att_context_size_all[i] = list(att_cs)
                if att_context_style == "chunked_limited":
                    if att_cs[0] > 0 and att_cs[0] % (att_cs[1] + 1) > 0:
                        raise ValueError(f"att_context_size[{i}][0] % (att_context_size[{i}][1] + 1) should be zero!")
                    if att_cs[1] < 0 and len(att_context_size_all) <= 1:
                        raise ValueError(
                            f"Right context (att_context_size[{i}][1]) can not be unlimited for chunked_limited style!"
                        )
        else:
            att_context_size_all = [[-1, -1]]

        if att_context_probs:
            if len(att_context_probs) != len(att_context_size_all):
                raise ValueError("The size of the att_context_probs should be the same as att_context_size.")
            att_context_probs = list(att_context_probs)
            if sum(att_context_probs) != 1:
                raise ValueError(
                    "The sum of numbers in att_context_probs should be equal to one to be a distribution."
                )
        else:
            att_context_probs = [1.0 / len(att_context_size_all)] * len(att_context_size_all)

        if conv_context_size is not None:
            if isinstance(conv_context_size, (list, tuple)):
                conv_context_size = list(conv_context_size)
            if not isinstance(conv_context_size, list) and not isinstance(conv_context_size, str):
                raise ValueError(
                    "Invalid conv_context_size! It should be the string 'causal' or a list of two integers."
                )
            if conv_context_size == "causal":
                conv_context_size = [conv_kernel_size - 1, 0]
            else:
                if conv_context_size[0] + conv_context_size[1] + 1 != conv_kernel_size:
                    raise ValueError(f"Invalid conv_context_size: {conv_context_size}!")
        else:
            conv_context_size = [(conv_kernel_size - 1) // 2, (conv_kernel_size - 1) // 2]
        return att_context_size_all, att_context_size_all[0], att_context_probs, conv_context_size
