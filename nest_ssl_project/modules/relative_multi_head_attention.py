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
Relative positional multi-head attention (matching NeMo implementation).
Based on Transformer-XL: https://arxiv.org/abs/1901.02860
"""

import math
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        super().__init__()
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_cache_len = max_cache_len
    
    def forward_qkv(self, query, key, value):
        """Transform query, key and value."""
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        return q, k, v
    
    def forward_attention(self, value, scores, mask):
        """Compute attention context vector."""
        n_batch = value.size(0)
        if mask is not None:
            # NeMo's mask format: True indicates positions to mask (padding positions)
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -INF_VAL)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)
    
    def forward(self, query, key, value, mask=None):
        """Compute scaled dot product attention."""
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k  # (batch, head, time1, time2)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    """
    
    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        pos_bias_u=None,
        pos_bias_v=None,
        max_cache_len=0,
        use_bias=True,
    ):
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
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v
    
    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = F.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x
    
    def update_cache(self, key, value, query, cache):
        """Update cache for streaming inference."""
        if cache is not None:
            key = value = torch.cat([cache, key], dim=1)
            q_keep_size = query.shape[1] - self.cache_drop_size if hasattr(self, 'cache_drop_size') else query.shape[1]
            cache = torch.cat([cache[:, q_keep_size:, :], query[:, :q_keep_size, :]], dim=1)
        return key, value, query, cache

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor): (batch, time1, size)
            cache (torch.Tensor): (batch, time_cache, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor): (batch, time_cache_next, size) if cache is not None
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            if pos_emb is None:
                # Fallback to standard attention if no positional encoding
                q, k, v = self.forward_qkv(query, key, value)
                scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
                out = self.forward_attention(v, scores, mask)
            else:
                q, k, v = self.forward_qkv(query, key, value)
                q = q.transpose(1, 2)  # (batch, time1, head, d_k)

                n_batch_pos = pos_emb.size(0)
                n_batch = value.size(0)
                p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
                p = p.transpose(1, 2)  # (batch, head, time1, d_k)

                # (batch, head, time1, d_k)
                q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
                # (batch, head, time1, d_k)
                q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

                # compute matrix b and matrix d
                # (batch, head, time1, time2)
                matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
                matrix_bd = self.rel_shift(matrix_bd)

                # drops extra elements in the matrix_bd to match the matrix_ac's size
                matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
                matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
                scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)

                out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache

