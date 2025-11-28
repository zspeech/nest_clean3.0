# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
ConformerEncoder implementation based on NeMo's implementation.
Simplified version for debugging, but follows NeMo's architecture closely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from core.classes.neural_module import NeuralModule
from core.classes.mixins import AccessMixin
from core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from modules.relative_multi_head_attention import RelPositionMultiHeadAttention
from modules.relative_positional_encoding import RelPositionalEncoding


class Swish(nn.SiLU):
    """Swish activation function (same as SiLU)."""
    pass


class ConformerFeedForward(nn.Module):
    """Feed-forward module of Conformer model (matches NeMo implementation)."""
    
    def __init__(self, d_model, d_ff, dropout, activation=Swish(), use_bias=True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ConformerConvolution(nn.Module):
    """Convolution module for Conformer (matches NeMo implementation)."""
    
    def __init__(
        self,
        d_model,
        kernel_size,
        norm_type='batch_norm',
        conv_context_size=None,
        use_bias=True,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        
        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2
        
        # Pointwise conv 1 (outputs 2*d_model for GLU)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )
        
        # Depthwise conv (after GLU, input is d_model channels)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,  # After GLU, channels are d_model
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=d_model,  # depthwise: groups = in_channels
            bias=use_bias,
        )
        
        # Normalization (after GLU, channels are d_model)
        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(d_model)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} not supported in simplified version")
        
        self.activation = Swish()
        
        # Pointwise conv 2
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )
    
    def forward(self, x, pad_mask=None, cache=None):
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        
        # GLU activation
        x = F.glu(x, dim=1)
        
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        
        x = self.depthwise_conv(x)
        
        # Normalization
        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)
        
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        
        # [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)
        
        return x


class ConformerLayer(nn.Module):
    """A single block of the Conformer encoder (matches NeMo structure)."""
    
    def __init__(
        self,
        d_model,
        d_ff,
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.1,
        use_bias=True,
        self_attention_model='rel_pos',
        pos_bias_u=None,
        pos_bias_v=None,
    ):
        super().__init__()
        self.fc_factor = 0.5  # Conformer uses 0.5 factor for FFN
        self.self_attention_model = self_attention_model
        
        # First feed-forward module
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)
        
        # Multi-headed self-attention module
        self.norm_self_att = nn.LayerNorm(d_model)
        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                use_bias=use_bias,
            )
        else:
            # Fallback to standard attention
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout_att)
        
        # Convolution module
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )
        
        # Second feed-forward module
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)
    
    def forward(self, x, att_mask=None, pad_mask=None, pos_emb=None):
        """
        Args:
            x: input signals (B, T, d_model)
            att_mask: attention masks (B, T, T) or None
            pad_mask: padding mask (B, T) or None
            pos_emb: positional embeddings (B, T, d_model) or None
        Returns:
            x: output (B, T, d_model)
        """
        residual = x
        
        # Feed-forward module 1
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor
        
        # Convolution module (NeMo's order: FFN1 -> Conv -> Attn -> FFN2)
        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)
        residual = residual + self.dropout(x)
        
        # Multi-head self-attention
        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos' and isinstance(self.self_attn, RelPositionMultiHeadAttention):
            attn_out = self.self_attn(x, x, x, mask=att_mask, pos_emb=pos_emb)
        else:
            attn_out, _ = self.self_attn(x, x, x, attn_mask=att_mask)
        residual = residual + self.dropout(attn_out)
        
        # Feed-forward module 2
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor
        
        # Final normalization
        x = self.norm_out(residual)
        
        return x


class ConformerPreEncoder(nn.Module):
    """
    Pre-encoder subsampling module matching NeMo's ConvSubsampling.
    Supports dw_striding subsampling with depthwise separable convolutions.
    """
    
    def __init__(
        self,
        feat_in: int,
        feat_out: int,
        subsampling: str = "dw_striding",
        subsampling_factor: int = 8,
        subsampling_conv_channels: int = 256,
        causal_downsampling: bool = False,
        dropout_pre_encoder: float = 0.1,
    ):
        super().__init__()
        import math
        
        self.subsampling_factor = subsampling_factor
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.is_causal = causal_downsampling
        
        if subsampling_factor % 2 != 0:
            raise ValueError("subsampling_factor should be a multiple of 2!")
        
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False
        
        if self.is_causal:
            self._left_padding = self._kernel_size - 1
            self._right_padding = self._stride - 1
        else:
            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2
        
        layers = []
        in_channels = 1
        
        if subsampling == "dw_striding":
            # Layer 1: First conv layer
            if self.is_causal:
                # For causal, we'd need CausalConv2D, but for simplicity use regular conv with padding
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=subsampling_conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=(self._left_padding, 0),  # Only pad left for causal
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=subsampling_conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
            layers.append(nn.ReLU())
            in_channels = subsampling_conv_channels
            
            # Additional layers: depthwise separable convolutions
            for i in range(self._sampling_num - 1):
                # Depthwise conv
                if self.is_causal:
                    layers.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=(self._left_padding, 0),
                            groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        )
                    )
                
                # Pointwise conv
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=subsampling_conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                layers.append(nn.ReLU())
                in_channels = subsampling_conv_channels
        else:
            # Fallback: simple striding
            for i in range(self._sampling_num):
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=subsampling_conv_channels if i < self._sampling_num - 1 else feat_out,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
                layers.append(nn.ReLU())
                in_channels = subsampling_conv_channels if i < self._sampling_num - 1 else feat_out
        
        self.conv = nn.Sequential(*layers)
        
        # Calculate output length for Linear layer (matching NeMo)
        if subsampling == "dw_striding":
            # Calculate output frequency dimension using NeMo's calc_length function
            # This matches NeMo's calculation: conv_channels * out_length
            def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num):
                """Calculate output length after conv layers."""
                add_pad = all_paddings - kernel_size
                one = 1.0
                for i in range(repeat_num):
                    lengths = (lengths + add_pad) / stride + one
                    if ceil_mode:
                        lengths = math.ceil(lengths)
                    else:
                        lengths = math.floor(lengths)
                return int(lengths)
            
            in_length = float(feat_in)
            all_paddings = self._left_padding + self._right_padding
            out_length = calc_length(
                in_length,
                all_paddings,
                self._kernel_size,
                self._stride,
                self._ceil_mode,
                self._sampling_num,
            )
            # Linear layer: (conv_channels * out_length) -> feat_out
            self.out = nn.Linear(subsampling_conv_channels * out_length, feat_out)
            self.conv2d_subsampling = True
        else:
            self.out = None
            self.conv2d_subsampling = False
        
        self.dropout = nn.Dropout(dropout_pre_encoder) if dropout_pre_encoder > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [B, D, T]
            lengths: Length tensor of shape [B]
        Returns:
            feats: Output features of shape [B, D', T']
            lengths: Updated lengths of shape [B]
        """
        # Add channel dimension: [B, D, T] -> [B, 1, D, T]
        x = x.unsqueeze(1)
        
        # Apply convolution: [B, 1, D, T] -> [B, C, H', T']
        feats = self.conv(x)
        
        if self.conv2d_subsampling:
            # Flatten Channel and Frequency Axes (matching NeMo)
            # [B, C, H', T'] -> [B, T', C*H']
            B, C, H, T = feats.shape
            feats = feats.transpose(1, 2).reshape(B, T, -1)  # [B, T, C*H]
            # Linear projection: [B, T, C*H] -> [B, T, feat_out]
            feats = self.out(feats)
            # Transpose to [B, feat_out, T] to match expected output format
            feats = feats.transpose(1, 2)  # [B, feat_out, T]
        else:
            # Remove height dimension: [B, D', H', T'] -> [B, D', T']
            B, C, H, T = feats.shape
            if H == 1:
                feats = feats.squeeze(2)  # [B, C, T]
            else:
                # Average over height dimension
                feats = feats.mean(dim=2)  # [B, C, T]
        
        # Apply dropout
        feats = self.dropout(feats)
        
        # Update lengths (each conv layer reduces by factor of 2)
        lengths = (lengths.float() / self.subsampling_factor).long()
        lengths = torch.clamp(lengths, min=1)
        
        return feats, lengths


class ConformerEncoder(NeuralModule, AccessMixin):
    """
    ConformerEncoder implementation matching NeMo's architecture.
    Based on: 'Conformer: Convolution-augmented Transformer for Speech Recognition'
    """
    
    @property
    def input_types(self):
        return {
            "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }
    
    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }
    
    def __init__(
        self,
        feat_in: int,
        feat_out: int = -1,
        n_layers: int = 17,
        d_model: int = 512,
        use_bias: bool = True,
        subsampling: str = "dw_striding",
        subsampling_factor: int = 8,
        subsampling_conv_channels: int = 256,
        causal_downsampling: bool = False,
        ff_expansion_factor: int = 4,
        self_attention_model: str = "rel_pos",
        n_heads: int = 8,
        att_context_size: Optional[list] = None,
        att_context_style: str = "regular",
        xscaling: bool = True,
        untie_biases: bool = True,
        pos_emb_max_len: int = 5000,
        conv_kernel_size: int = 9,
        conv_norm_type: str = "batch_norm",
        conv_context_size: Optional[list] = None,
        dropout: float = 0.1,
        dropout_pre_encoder: float = 0.1,
        dropout_emb: float = 0.0,
        dropout_att: float = 0.1,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        **kwargs
    ):
        super().__init__()
        
        self.feat_in = feat_in
        self.feat_out = feat_out if feat_out > 0 else d_model
        self.d_model = d_model
        self.n_layers = n_layers
        self.subsampling_factor = subsampling_factor
        self.self_attention_model = self_attention_model
        d_ff = d_model * ff_expansion_factor
        
        # X-scaling
        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None
        
        # Pre-encoder (subsampling)
        self.pre_encode = ConformerPreEncoder(
            feat_in=feat_in,
            feat_out=d_model,
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            subsampling_conv_channels=subsampling_conv_channels,
            causal_downsampling=causal_downsampling,
            dropout_pre_encoder=dropout_pre_encoder,
        )
        
        # Positional encoding
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            # Biases for relative positional encoding
            # If untie_biases=False, share biases across layers; if True, each layer has its own biases
            if not untie_biases:
                d_head = d_model // n_heads
                pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
                pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
                nn.init.zeros_(pos_bias_u)
                nn.init.zeros_(pos_bias_v)
            else:
                pos_bias_u = None
                pos_bias_v = None
            
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        else:
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = None
        
        # Conformer layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # If untie_biases=True, each layer gets its own bias parameters
            # If untie_biases=False, all layers share the same bias parameters
            layer_pos_bias_u = None if untie_biases else pos_bias_u
            layer_pos_bias_v = None if untie_biases else pos_bias_v
            
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                conv_context_size=conv_context_size,
                dropout=dropout,
                dropout_att=dropout_att,
                use_bias=use_bias,
                self_attention_model=self_attention_model,
                pos_bias_u=layer_pos_bias_u,
                pos_bias_v=layer_pos_bias_v,
            )
            self.layers.append(layer)
        
        # Output projection
        if self.feat_out != d_model:
            self.output_proj = nn.Linear(d_model, self.feat_out)
        else:
            self.output_proj = nn.Identity()
        
        # Initialize AccessMixin
        self._setup_access_mixin()
    
    def _setup_access_mixin(self):
        """Setup AccessMixin for layer access."""
        self._access_enabled = False
        self._access_cfg = {}
        self._registry = {}
    
    def reset_registry(self):
        """Reset the access registry."""
        self._registry = {}
    
    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        cache_last_channel: Optional[torch.Tensor] = None,
        cache_last_time: Optional[torch.Tensor] = None,
        cache_last_channel_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            audio_signal: Input tensor of shape [B, D, T]
            length: Length tensor of shape [B]
            cache_last_channel: Optional cache (not used in simplified version)
            cache_last_time: Optional cache (not used in simplified version)
            cache_last_channel_len: Optional cache (not used in simplified version)
        
        Returns:
            encoded: Encoded features of shape [B, D', T']
            encoded_len: Updated lengths of shape [B]
        """
        # Pre-encode (subsampling)
        x, lengths = self.pre_encode(audio_signal, length)
        
        # Transpose for transformer: [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)
        
        # Positional encoding
        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(x)
            # pos_emb shape: (1, T, d_model) -> expand to (B, T, d_model)
            if pos_emb.size(0) == 1:
                pos_emb = pos_emb.expand(x.size(0), -1, -1)
        else:
            pos_emb = None
        
        # Apply Conformer layers
        for layer in self.layers:
            x = layer(x, pos_emb=pos_emb)
        
        # Output projection
        x = self.output_proj(x)
        
        # Transpose back: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        
        return x, lengths
