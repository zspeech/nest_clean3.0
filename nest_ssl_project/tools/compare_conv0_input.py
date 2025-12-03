#!/usr/bin/env python3
"""
Compare encoder.pre_encode.conv.0 input between NeMo and nest.
"""

import torch
import pickle
from pathlib import Path


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("COMPARE encoder.pre_encode.conv.0 INPUT")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get conv.0 inputs
    nemo_conv0 = nemo_layers.get('encoder.pre_encode.conv.0', {})
    nest_conv0 = nest_layers.get('encoder.pre_encode.conv.0', {})
    
    nemo_inputs = nemo_conv0.get('all_forward_inputs', [])
    nest_inputs = nest_conv0.get('all_forward_inputs', [])
    
    if len(nemo_inputs) == 0 or len(nest_inputs) == 0:
        print("ERROR: No conv.0 inputs captured")
        return
    
    nemo_input = nemo_inputs[0]
    nest_input = nest_inputs[0]
    
    if isinstance(nemo_input, (list, tuple)) and len(nemo_input) > 0:
        nemo_input_tensor = nemo_input[0]
    else:
        nemo_input_tensor = nemo_input
    
    if isinstance(nest_input, (list, tuple)) and len(nest_input) > 0:
        nest_input_tensor = nest_input[0]
    else:
        nest_input_tensor = nest_input
    
    print(f"\nNeMo conv.0 input:")
    print(f"  shape: {nemo_input_tensor.shape}")
    print(f"  mean: {nemo_input_tensor.float().mean():.6f}")
    print(f"  min: {nemo_input_tensor.float().min():.6f}")
    print(f"  max: {nemo_input_tensor.float().max():.6f}")
    
    print(f"\nnest conv.0 input:")
    print(f"  shape: {nest_input_tensor.shape}")
    print(f"  mean: {nest_input_tensor.float().mean():.6f}")
    print(f"  min: {nest_input_tensor.float().min():.6f}")
    print(f"  max: {nest_input_tensor.float().max():.6f}")
    
    # Compare
    if nemo_input_tensor.shape == nest_input_tensor.shape:
        diff = (nemo_input_tensor.float() - nest_input_tensor.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nComparison:")
        print(f"  max_diff: {max_diff:.6e}")
        print(f"  mean_diff: {mean_diff:.6e}")
        
        if max_diff < 1e-5:
            print(f"  [OK] Perfect match!")
        else:
            print(f"  [FAIL] Mismatch detected")
            # Find location of max diff
            max_diff_idx = diff.argmax().item()
            max_diff_coords = []
            for dim_size in reversed(nemo_input_tensor.shape):
                max_diff_coords.append(max_diff_idx % dim_size)
                max_diff_idx //= dim_size
            max_diff_coords.reverse()
            print(f"  Max diff at: {max_diff_coords}")
            print(f"  NeMo value: {nemo_input_tensor.flatten()[diff.argmax()].item():.6f}")
            print(f"  nest value: {nest_input_tensor.flatten()[diff.argmax()].item():.6f}")
    else:
        print(f"\n[FAIL] Shape mismatch:")
        print(f"  NeMo: {nemo_input_tensor.shape}")
        print(f"  nest: {nest_input_tensor.shape}")


if __name__ == '__main__':
    main()

