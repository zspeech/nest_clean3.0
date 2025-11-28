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
Relative positional encoding for Transformer-XL (matching NeMo implementation).
"""

import math
import torch
import torch.nn as nn

INF_VAL = 10000.0


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""
    
    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None
    
    def create_pe(self, positions, dtype):
        """Create positional encoding."""
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
        self.extend_pe(input_len, x.device, x.dtype)
        pos_emb = self.pe[:, :input_len]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for Transformer-XL.
    See: Appendix B in https://arxiv.org/abs/1901.02860
    """
    
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
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        if self.xscale:
            x = x * self.xscale
        
        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = x.size(1) + cache_len
        self.extend_pe(input_len, x.device, x.dtype)
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb

