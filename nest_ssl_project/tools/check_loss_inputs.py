#!/usr/bin/env python3
"""
Check mask and target values to diagnose NaN loss.
"""

import torch
import pickle
from pathlib import Path

def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK LOSS INPUTS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Check quantizer (tokens/targets)
    nemo_quant = nemo_layers.get('quantizer', {})
    nest_quant = nest_layers.get('quantizer', {})
    
    nemo_quant_outs = nemo_quant.get('all_forward_outputs', [])
    nest_quant_outs = nest_quant.get('all_forward_outputs', [])
    
    if len(nemo_quant_outs) > 0 and len(nest_quant_outs) > 0:
        nemo_tokens = nemo_quant_outs[0][1] if isinstance(nemo_quant_outs[0], (list, tuple)) else None
        nest_tokens = nest_quant_outs[0][1] if isinstance(nest_quant_outs[0], (list, tuple)) else None
        
        if isinstance(nemo_tokens, torch.Tensor):
            print(f"\nNeMo tokens shape: {nemo_tokens.shape}")
            print(f"NeMo tokens min: {nemo_tokens.min()}, max: {nemo_tokens.max()}")
            print(f"NeMo tokens sample: {nemo_tokens[0, :10]}")
        
        if isinstance(nest_tokens, torch.Tensor):
            print(f"\nnest tokens shape: {nest_tokens.shape}")
            print(f"nest tokens min: {nest_tokens.min()}, max: {nest_tokens.max()}")
            print(f"nest tokens sample: {nest_tokens[0, :10]}")
    
    # Check mask_processor output (masks)
    nemo_mask_proc = nemo_layers.get('mask_processor', {})
    nest_mask_proc = nest_layers.get('mask_processor', {})
    
    nemo_mask_outs = nemo_mask_proc.get('all_forward_outputs', [])
    nest_mask_outs = nest_mask_proc.get('all_forward_outputs', [])
    
    if len(nemo_mask_outs) > 0 and len(nest_mask_outs) > 0:
        nemo_masks = nemo_mask_outs[0][1] if isinstance(nemo_mask_outs[0], (list, tuple)) else None
        nest_masks = nest_mask_outs[0][1] if isinstance(nest_mask_outs[0], (list, tuple)) else None
        
        if isinstance(nemo_masks, torch.Tensor):
            print(f"\nNeMo masks shape: {nemo_masks.shape}")
            print(f"NeMo masks min: {nemo_masks.min()}, max: {nemo_masks.max()}")
            print(f"NeMo masks sum (total masked): {nemo_masks.sum()}")
            print(f"NeMo masks mean: {nemo_masks.float().mean()}")
            # Check if masks are all zeros
            if nemo_masks.sum() == 0:
                print("  [WARNING] NeMo masks are all zeros!")
            else:
                print(f"  NeMo has {nemo_masks.sum().item()} masked positions")
        
        if isinstance(nest_masks, torch.Tensor):
            print(f"\nnest masks shape: {nest_masks.shape}")
            print(f"nest masks min: {nest_masks.min()}, max: {nest_masks.max()}")
            print(f"nest masks sum (total masked): {nest_masks.sum()}")
            print(f"nest masks mean: {nest_masks.float().mean()}")
            if nest_masks.sum() == 0:
                print("  [WARNING] nest masks are all zeros!")
            else:
                print(f"  nest has {nest_masks.sum().item()} masked positions")
        
        # Compare masks
        if isinstance(nemo_masks, torch.Tensor) and isinstance(nest_masks, torch.Tensor):
            if nemo_masks.shape == nest_masks.shape:
                diff = (nemo_masks.float() - nest_masks.float()).abs().max().item()
                print(f"\nMask comparison: max_diff={diff:.6e} {'[OK]' if diff < 1e-5 else '[FAIL]'}")
    
    # Check decoder output (log_probs)
    nemo_decoder = nemo_layers.get('decoder', {})
    nest_decoder = nest_layers.get('decoder', {})
    
    nemo_decoder_outs = nemo_decoder.get('all_forward_outputs', [])
    nest_decoder_outs = nest_decoder.get('all_forward_outputs', [])
    
    if len(nemo_decoder_outs) > 0 and len(nest_decoder_outs) > 0:
        nemo_logprobs = nemo_decoder_outs[0]
        nest_logprobs = nest_decoder_outs[0]
        
        if isinstance(nemo_logprobs, torch.Tensor):
            print(f"\nNeMo decoder output shape: {nemo_logprobs.shape}")
            print(f"NeMo decoder output min: {nemo_logprobs.min()}, max: {nemo_logprobs.max()}")
            print(f"NeMo decoder output has NaN: {torch.isnan(nemo_logprobs).any()}")
            print(f"NeMo decoder output has Inf: {torch.isinf(nemo_logprobs).any()}")
        
        if isinstance(nest_logprobs, torch.Tensor):
            print(f"\nnest decoder output shape: {nest_logprobs.shape}")
            print(f"nest decoder output min: {nest_logprobs.min()}, max: {nest_logprobs.max()}")
            print(f"nest decoder output has NaN: {torch.isnan(nest_logprobs).any()}")
            print(f"nest decoder output has Inf: {torch.isinf(nest_logprobs).any()}")

if __name__ == '__main__':
    main()

