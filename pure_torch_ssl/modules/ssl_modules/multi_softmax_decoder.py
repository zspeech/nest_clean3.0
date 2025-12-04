# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Simplified for standalone use.

import torch
import torch.nn as nn


def init_weights(m, mode='xavier_uniform'):
    """Initialize weights."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class MultiSoftmaxDecoder(nn.Module):
    """Multi-head softmax decoder for SSL."""

    def __init__(self, feat_in: int, num_classes: int, num_decoders: int = 1,
                 init_mode: str = "xavier_uniform", use_bias: bool = False, squeeze_single: bool = False):
        super().__init__()
        self.feat_in = feat_in
        self.num_classes = num_classes
        self.num_decoders = num_decoders
        self.squeeze_single = squeeze_single

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self.feat_in, self.num_classes * self.num_decoders, kernel_size=1, bias=use_bias)
        )
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encoder_output):
        """Forward pass. Input: (B, D, T), Output: (B, T, C, H) or (B, T, C)."""
        logits = self.decoder_layers(encoder_output).transpose(1, 2)
        logits = logits.reshape(logits.shape[0], logits.shape[1], self.num_classes, self.num_decoders)
        if self.squeeze_single and self.num_decoders == 1:
            logits = logits.squeeze(-1)
        return torch.nn.functional.log_softmax(logits, dim=2)
