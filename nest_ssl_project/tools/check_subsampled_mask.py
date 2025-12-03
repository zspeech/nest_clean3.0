#!/usr/bin/env python3
"""
Simulate MLMLoss mask subsampling to check if mask is valid after subsampling.
"""

import torch
import pickle
from pathlib import Path

def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK SUBSAMPLED MASK")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get masks from mask_processor
    nemo_mask_proc = nemo_layers.get('mask_processor', {})
    nest_mask_proc = nest_layers.get('mask_processor', {})
    
    nemo_mask_outs = nemo_mask_proc.get('all_forward_outputs', [])
    nest_mask_outs = nest_mask_proc.get('all_forward_outputs', [])
    
    if len(nemo_mask_outs) > 0 and len(nest_mask_outs) > 0:
        nemo_masks = nemo_mask_outs[0][1] if isinstance(nemo_mask_outs[0], (list, tuple)) else None
        nest_masks = nest_mask_outs[0][1] if isinstance(nest_mask_outs[0], (list, tuple)) else None
        
        if isinstance(nemo_masks, torch.Tensor) and isinstance(nest_masks, torch.Tensor):
            print(f"\nOriginal masks shape: {nemo_masks.shape}")
            print(f"NeMo masks sum: {nemo_masks.sum()}")
            print(f"nest masks sum: {nest_masks.sum()}")
            
            # Simulate MLMLoss processing
            combine_time_steps = 8  # subsampling_factor
            mask_threshold = 0.8
            
            # B,D,T -> B,T,D
            nemo_masks_t = nemo_masks.transpose(1, 2)  # [2, 1584, 80]
            nest_masks_t = nest_masks.transpose(1, 2)  # [2, 1584, 80]
            
            print(f"\nAfter transpose: shape={nemo_masks_t.shape}")
            
            # Reshape: [B, T // combine_time_steps, combine_time_steps * D]
            nemo_masks_reshaped = nemo_masks_t.reshape(
                nemo_masks_t.shape[0], 
                nemo_masks_t.shape[1] // combine_time_steps, 
                -1
            )  # [2, 198, 640]
            nest_masks_reshaped = nest_masks_t.reshape(
                nest_masks_t.shape[0], 
                nest_masks_t.shape[1] // combine_time_steps, 
                -1
            )  # [2, 198, 640]
            
            print(f"After reshape: shape={nemo_masks_reshaped.shape}")
            
            # Mean over frequency dimension
            nemo_masks_mean = nemo_masks_reshaped.mean(-1)  # [2, 198]
            nest_masks_mean = nest_masks_reshaped.mean(-1)  # [2, 198]
            
            print(f"After mean: shape={nemo_masks_mean.shape}")
            print(f"NeMo masks mean min: {nemo_masks_mean.min()}, max: {nemo_masks_mean.max()}")
            print(f"nest masks mean min: {nest_masks_mean.min()}, max: {nest_masks_mean.max()}")
            
            # Apply threshold
            nemo_masks_bool = nemo_masks_mean > mask_threshold  # [2, 198]
            nest_masks_bool = nest_masks_mean > mask_threshold  # [2, 198]
            
            print(f"\nAfter threshold (> {mask_threshold}):")
            print(f"NeMo masks sum (masked positions): {nemo_masks_bool.sum()}")
            print(f"nest masks sum (masked positions): {nest_masks_bool.sum()}")
            
            if nemo_masks_bool.sum() == 0:
                print("  [WARNING] NeMo has NO masked positions! This will cause NaN loss!")
            if nest_masks_bool.sum() == 0:
                print("  [WARNING] nest has NO masked positions! This will cause NaN loss!")
            
            # Check which frames are masked
            nemo_masked_frames = torch.where(nemo_masks_bool[0])[0]
            nest_masked_frames = torch.where(nest_masks_bool[0])[0]
            
            print(f"\nNeMo masked frames (sample 0): {nemo_masked_frames.tolist()}")
            print(f"nest masked frames (sample 0): {nest_masked_frames.tolist()}")

if __name__ == '__main__':
    main()

