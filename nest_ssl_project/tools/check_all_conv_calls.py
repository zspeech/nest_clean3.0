#!/usr/bin/env python3
"""
Check all calls to pre_encode.conv to see if split path is taken.
"""

import torch
import pickle
from pathlib import Path


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK ALL CONV CALLS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get all conv inputs
    nemo_conv_inputs = nemo_layers.get('encoder.pre_encode.conv', {}).get('all_forward_inputs', [])
    nest_conv_inputs = nest_layers.get('encoder.pre_encode.conv', {}).get('all_forward_inputs', [])
    
    print(f"\nNeMo pre_encode.conv calls: {len(nemo_conv_inputs)}")
    print(f"nest pre_encode.conv calls: {len(nest_conv_inputs)}")
    
    # Get preprocessor Call 1 output
    nemo_preproc_outs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preproc_outs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    
    nemo_preproc_call1 = nemo_preproc_outs[1][0]  # Call 1, output[0]
    nest_preproc_call1 = nest_preproc_outs[1][0]  # Call 1, output[0]
    
    # Simulate: transpose then unsqueeze
    nemo_expected = nemo_preproc_call1.transpose(1, 2).unsqueeze(1)  # [B, 1, T, D]
    nest_expected = nest_preproc_call1.transpose(1, 2).unsqueeze(1)  # [B, 1, T, D]
    
    print(f"\nExpected shape (after transpose + unsqueeze):")
    print(f"  NeMo: {nemo_expected.shape}")
    print(f"  nest: {nest_expected.shape}")
    
    # Check each conv call
    print("\n" + "="*80)
    print("NeMo conv calls:")
    print("="*80)
    for i, conv_input in enumerate(nemo_conv_inputs):
        if isinstance(conv_input, (list, tuple)) and len(conv_input) > 0:
            conv_input_tensor = conv_input[0]
        else:
            conv_input_tensor = conv_input
        
        if isinstance(conv_input_tensor, torch.Tensor):
            print(f"\nCall {i}:")
            print(f"  shape: {conv_input_tensor.shape}")
            print(f"  mean: {conv_input_tensor.float().mean():.6f}")
            
            # Compare with expected
            if conv_input_tensor.shape == nemo_expected.shape:
                diff = (conv_input_tensor - nemo_expected).abs()
                max_diff = diff.max().item()
                print(f"  vs expected: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
            else:
                print(f"  vs expected: shape mismatch (expected {nemo_expected.shape})")
                # Check if it's a chunk
                if conv_input_tensor.shape[0] < nemo_expected.shape[0]:
                    print(f"  [SPLIT] This appears to be a batch chunk (batch_size={conv_input_tensor.shape[0]} vs {nemo_expected.shape[0]})")
                elif conv_input_tensor.shape[2] < nemo_expected.shape[2]:
                    print(f"  [SPLIT] This appears to be a time chunk (time={conv_input_tensor.shape[2]} vs {nemo_expected.shape[2]})")
    
    print("\n" + "="*80)
    print("nest conv calls:")
    print("="*80)
    for i, conv_input in enumerate(nest_conv_inputs):
        if isinstance(conv_input, (list, tuple)) and len(conv_input) > 0:
            conv_input_tensor = conv_input[0]
        else:
            conv_input_tensor = conv_input
        
        if isinstance(conv_input_tensor, torch.Tensor):
            print(f"\nCall {i}:")
            print(f"  shape: {conv_input_tensor.shape}")
            print(f"  mean: {conv_input_tensor.float().mean():.6f}")
            
            # Compare with expected
            if conv_input_tensor.shape == nest_expected.shape:
                diff = (conv_input_tensor - nest_expected).abs()
                max_diff = diff.max().item()
                print(f"  vs expected: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
            else:
                print(f"  vs expected: shape mismatch (expected {nest_expected.shape})")
                # Check if it's a chunk
                if conv_input_tensor.shape[0] < nest_expected.shape[0]:
                    print(f"  [SPLIT] This appears to be a batch chunk (batch_size={conv_input_tensor.shape[0]} vs {nest_expected.shape[0]})")
                elif conv_input_tensor.shape[2] < nest_expected.shape[2]:
                    print(f"  [SPLIT] This appears to be a time chunk (time={conv_input_tensor.shape[2]} vs {nest_expected.shape[2]})")
    
    # Compare NeMo vs nest conv calls
    if len(nemo_conv_inputs) > 0 and len(nest_conv_inputs) > 0:
        print("\n" + "="*80)
        print("NeMo vs nest conv calls comparison:")
        print("="*80)
        
        nemo_first = nemo_conv_inputs[0]
        nest_first = nest_conv_inputs[0]
        
        if isinstance(nemo_first, (list, tuple)) and len(nemo_first) > 0:
            nemo_first_tensor = nemo_first[0]
        else:
            nemo_first_tensor = nemo_first
        
        if isinstance(nest_first, (list, tuple)) and len(nest_first) > 0:
            nest_first_tensor = nest_first[0]
        else:
            nest_first_tensor = nest_first
        
        if isinstance(nemo_first_tensor, torch.Tensor) and isinstance(nest_first_tensor, torch.Tensor):
            if nemo_first_tensor.shape == nest_first_tensor.shape:
                diff = (nemo_first_tensor - nest_first_tensor).abs()
                max_diff = diff.max().item()
                print(f"\nFirst call: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
            else:
                print(f"\nFirst call: shape mismatch")
                print(f"  NeMo: {nemo_first_tensor.shape}")
                print(f"  nest: {nest_first_tensor.shape}")


if __name__ == '__main__':
    main()

