#!/usr/bin/env python3
"""
Compare preprocessor Call 1 output with pre_encode.conv input.
"""

import torch
import pickle
from pathlib import Path


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("PREPROCESSOR CALL 1 -> PRE_ENCODE.CONV INPUT COMPARISON")
    print("="*80)
    
    # Load layer outputs
    print("\nLoading NeMo layer outputs...")
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    print("Loading nest layer outputs...")
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get preprocessor Call 1 output
    nemo_preproc_outs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preproc_outs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    
    if len(nemo_preproc_outs) < 2 or len(nest_preproc_outs) < 2:
        print("ERROR: Not enough preprocessor calls")
        return
    
    nemo_preproc_call1 = nemo_preproc_outs[1][0]  # Call 1, output[0]
    nest_preproc_call1 = nest_preproc_outs[1][0]  # Call 1, output[0]
    
    print(f"\nPreprocessor Call 1 output:")
    print(f"  NeMo: shape={nemo_preproc_call1.shape}, mean={nemo_preproc_call1.float().mean():.6f}")
    print(f"  nest: shape={nest_preproc_call1.shape}, mean={nest_preproc_call1.float().mean():.6f}")
    
    # Compare preprocessor Call 1 outputs
    diff_preproc = (nemo_preproc_call1 - nest_preproc_call1).abs()
    max_diff_preproc = diff_preproc.max().item()
    print(f"\nPreprocessor Call 1 output comparison:")
    print(f"  max_diff={max_diff_preproc:.6e} {'[OK]' if max_diff_preproc < 1e-5 else '[FAIL]'}")
    
    # Get pre_encode.conv input
    nemo_conv_inputs = nemo_layers.get('encoder.pre_encode.conv', {}).get('all_forward_inputs', [])
    nest_conv_inputs = nest_layers.get('encoder.pre_encode.conv', {}).get('all_forward_inputs', [])
    
    if len(nemo_conv_inputs) == 0 or len(nest_conv_inputs) == 0:
        print("\nERROR: No conv inputs captured")
        return
    
    nemo_conv_input = nemo_conv_inputs[0][0] if isinstance(nemo_conv_inputs[0], (list, tuple)) else nemo_conv_inputs[0]
    nest_conv_input = nest_conv_inputs[0][0] if isinstance(nest_conv_inputs[0], (list, tuple)) else nest_conv_inputs[0]
    
    print(f"\npre_encode.conv input:")
    print(f"  NeMo: shape={nemo_conv_input.shape}, mean={nemo_conv_input.float().mean():.6f}")
    print(f"  nest: shape={nest_conv_input.shape}, mean={nest_conv_input.float().mean():.6f}")
    
    # Compare conv inputs
    diff_conv = (nemo_conv_input - nest_conv_input).abs()
    max_diff_conv = diff_conv.max().item()
    print(f"\npre_encode.conv input comparison:")
    print(f"  max_diff={max_diff_conv:.6e} {'[OK]' if max_diff_conv < 1e-5 else '[FAIL]'}")
    
    # Simulate transformation: transpose(1, 2) then unsqueeze(1)
    print("\n" + "="*80)
    print("SIMULATED TRANSFORMATION")
    print("="*80)
    
    # Step 1: transpose(1, 2): [B, D, T] -> [B, T, D]
    nemo_transposed = nemo_preproc_call1.transpose(1, 2)
    nest_transposed = nest_preproc_call1.transpose(1, 2)
    
    print(f"\nAfter transpose(1, 2):")
    print(f"  NeMo: shape={nemo_transposed.shape}, mean={nemo_transposed.float().mean():.6f}")
    print(f"  nest: shape={nest_transposed.shape}, mean={nest_transposed.float().mean():.6f}")
    
    diff_transposed = (nemo_transposed - nest_transposed).abs()
    max_diff_transposed = diff_transposed.max().item()
    print(f"  Comparison: max_diff={max_diff_transposed:.6e} {'[OK]' if max_diff_transposed < 1e-5 else '[FAIL]'}")
    
    # Step 2: unsqueeze(1): [B, T, D] -> [B, 1, T, D]
    nemo_unsqueezed = nemo_transposed.unsqueeze(1)
    nest_unsqueezed = nest_transposed.unsqueeze(1)
    
    print(f"\nAfter unsqueeze(1):")
    print(f"  NeMo: shape={nemo_unsqueezed.shape}, mean={nemo_unsqueezed.float().mean():.6f}")
    print(f"  nest: shape={nest_unsqueezed.shape}, mean={nest_unsqueezed.float().mean():.6f}")
    
    diff_unsqueezed = (nemo_unsqueezed - nest_unsqueezed).abs()
    max_diff_unsqueezed = diff_unsqueezed.max().item()
    print(f"  Comparison: max_diff={max_diff_unsqueezed:.6e} {'[OK]' if max_diff_unsqueezed < 1e-5 else '[FAIL]'}")
    
    # Compare simulated vs actual conv input
    print("\n" + "="*80)
    print("SIMULATED VS ACTUAL CONV INPUT")
    print("="*80)
    
    diff_nemo_sim = (nemo_unsqueezed - nemo_conv_input).abs()
    max_diff_nemo_sim = diff_nemo_sim.max().item()
    print(f"\nNeMo: simulated vs actual:")
    print(f"  max_diff={max_diff_nemo_sim:.6e} {'[OK]' if max_diff_nemo_sim < 1e-5 else '[FAIL]'}")
    if max_diff_nemo_sim > 1e-5:
        print(f"  Mean diff={diff_nemo_sim.mean().item():.6e}")
        print(f"  Max diff location: {diff_nemo_sim.argmax().item()}")
    
    diff_nest_sim = (nest_unsqueezed - nest_conv_input).abs()
    max_diff_nest_sim = diff_nest_sim.max().item()
    print(f"\nnest: simulated vs actual:")
    print(f"  max_diff={max_diff_nest_sim:.6e} {'[OK]' if max_diff_nest_sim < 1e-5 else '[FAIL]'}")
    if max_diff_nest_sim > 1e-5:
        print(f"  Mean diff={diff_nest_sim.mean().item():.6e}")
        print(f"  Max diff location: {diff_nest_sim.argmax().item()}")


if __name__ == '__main__':
    main()

