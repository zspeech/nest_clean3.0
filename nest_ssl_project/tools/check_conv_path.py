#!/usr/bin/env python3
"""
Check if ConvSubsampling takes the split path or direct path.
"""

import torch
import pickle
from pathlib import Path
import math


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK CONV PATH")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get preprocessor Call 1 output
    nemo_preproc_outs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preproc_outs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    
    nemo_preproc_call1 = nemo_preproc_outs[1][0]  # Call 1, output[0]
    nest_preproc_call1 = nest_preproc_outs[1][0]  # Call 1, output[0]
    
    # Simulate: transpose then unsqueeze
    nemo_x = nemo_preproc_call1.transpose(1, 2).unsqueeze(1)  # [B, 1, T, D]
    nest_x = nest_preproc_call1.transpose(1, 2).unsqueeze(1)  # [B, 1, T, D]
    
    print(f"\nAfter transpose + unsqueeze:")
    print(f"  NeMo: shape={nemo_x.shape}, numel={nemo_x.numel()}")
    print(f"  nest: shape={nest_x.shape}, numel={nest_x.numel()}")
    
    # Check if split is needed (assuming conv2d_subsampling=True, subsampling_conv_chunking_factor=1)
    # From ConvSubsampling.forward logic:
    # if subsampling_conv_chunking_factor == 1:
    #     x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
    #     if torch.numel(x) > x_ceil:
    #         need_to_split = True
    
    # Typical values from config:
    conv_channels = 256  # typical value
    stride = 2  # typical value
    
    x_ceil = 2**31 / conv_channels * stride * stride
    print(f"\nSplit threshold (x_ceil): {x_ceil:.0f}")
    
    nemo_need_split = nemo_x.numel() > x_ceil
    nest_need_split = nest_x.numel() > x_ceil
    
    print(f"\nNeMo: need_to_split = {nemo_need_split}")
    print(f"nest: need_to_split = {nest_need_split}")
    
    # Get actual conv input
    nemo_conv_inputs = nemo_layers.get('encoder.pre_encode.conv', {}).get('all_forward_inputs', [])
    nest_conv_inputs = nest_layers.get('encoder.pre_encode.conv', {}).get('all_forward_inputs', [])
    
    if len(nemo_conv_inputs) > 0 and len(nest_conv_inputs) > 0:
        nemo_conv_input = nemo_conv_inputs[0][0] if isinstance(nemo_conv_inputs[0], (list, tuple)) else nemo_conv_inputs[0]
        nest_conv_input = nest_conv_inputs[0][0] if isinstance(nest_conv_inputs[0], (list, tuple)) else nest_conv_inputs[0]
        
        print(f"\nActual conv input:")
        print(f"  NeMo: shape={nemo_conv_input.shape}, numel={nemo_conv_input.numel()}")
        print(f"  nest: shape={nest_conv_input.shape}, numel={nest_conv_input.numel()}")
        
        # Check if shape changed (indicating split)
        nemo_shape_changed = nemo_conv_input.shape != nemo_x.shape
        nest_shape_changed = nest_conv_input.shape != nest_x.shape
        
        print(f"\nShape changed (indicating split):")
        print(f"  NeMo: {nemo_shape_changed}")
        print(f"  nest: {nest_shape_changed}")
        
        # Check batch size
        if nemo_shape_changed:
            print(f"\nNeMo: Original batch={nemo_x.shape[0]}, Conv input batch={nemo_conv_input.shape[0]}")
        if nest_shape_changed:
            print(f"nest: Original batch={nest_x.shape[0]}, Conv input batch={nest_conv_input.shape[0]}")


if __name__ == '__main__':
    main()

