# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# Simplified for standalone use - no type checking.

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MLMLoss", "MultiMLMLoss"]


class MLMLoss(nn.Module):
    """Masked Language Model Loss."""
    
    def __init__(self, combine_time_steps: int = 1, mask_threshold: float = 0.8):
        super().__init__()
        self.nll_loss = nn.NLLLoss()
        self.combine_time_steps = combine_time_steps
        self.mask_threshold = mask_threshold

    def forward(self, decoder_outputs, targets, decoder_lengths=None, target_lengths=None, spec_masks=None, masks=None):
        if masks is None:
            masks = spec_masks

        # B,D,T -> B,T,D
        masks = masks.transpose(1, 2)
        masks = masks.reshape(masks.shape[0], masks.shape[1] // self.combine_time_steps, -1)
        masks = masks.mean(-1) > self.mask_threshold

        # Return 0 loss if no masked positions (avoid NaN)
        if masks.sum().item() == 0:
            return torch.tensor(0.0, device=decoder_outputs.device, requires_grad=True)

        out_masked_only = decoder_outputs[masks]
        targets = F.pad(targets, (0, masks.shape[-1] - targets.shape[-1]))
        targets_masked_only = targets[masks]

        loss = self.nll_loss(out_masked_only, targets_masked_only)
        return torch.mean(loss)


class MultiMLMLoss(nn.Module):
    """Multi-decoder MLM Loss."""
    
    def __init__(self, combine_time_steps: int = 1, mask_threshold: float = 0.8, 
                 num_decoders: int = 1, squeeze_single: bool = False):
        super().__init__()
        self.num_decoders = num_decoders
        self.squeeze_single = squeeze_single
        self.mlm_loss = MLMLoss(combine_time_steps, mask_threshold)

    def forward(self, masks, decoder_outputs, targets, decoder_lengths=None, target_lengths=None):
        if self.squeeze_single and self.num_decoders == 1:
            return self.mlm_loss(spec_masks=masks, decoder_outputs=decoder_outputs, 
                                targets=targets, decoder_lengths=decoder_lengths, target_lengths=target_lengths)
        
        loss = 0.0
        for i in range(self.num_decoders):
            loss += self.mlm_loss(spec_masks=masks, decoder_outputs=decoder_outputs[:, :, :, i],
                                 targets=targets[:, :, i], decoder_lengths=decoder_lengths, target_lengths=target_lengths)
        return loss / self.num_decoders
