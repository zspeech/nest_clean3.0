#!/usr/bin/env python3
"""
Check the input to MaskedConvSequential (encoder.pre_encode.conv).
"""

import torch
import pickle
from pathlib import Path


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK MASKED CONV SEQUENTIAL INPUT")
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
    
    print(f"\nPreprocessor Call 1 output:")
    print(f"  NeMo: shape={nemo_preproc_call1.shape}")
    print(f"  nest: shape={nest_preproc_call1.shape}")
    
    # Simulate: transpose(1, 2) to get [B, T, D]
    nemo_transposed = nemo_preproc_call1.transpose(1, 2)  # [B, T, D]
    nest_transposed = nest_preproc_call1.transpose(1, 2)  # [B, T, D]
    
    print(f"\nAfter transpose(1, 2) (expected MaskedConvSequential input):")
    print(f"  NeMo: shape={nemo_transposed.shape}")
    print(f"  nest: shape={nest_transposed.shape}")
    
    # Get MaskedConvSequential input (from encoder.pre_encode.conv hook)
    nemo_conv = nemo_layers.get('encoder.pre_encode.conv', {})
    nest_conv = nest_layers.get('encoder.pre_encode.conv', {})
    
    nemo_conv_inputs = nemo_conv.get('all_forward_inputs', [])
    nest_conv_inputs = nest_conv.get('all_forward_inputs', [])
    
    print(f"\nActual MaskedConvSequential input (from hook):")
    if len(nemo_conv_inputs) > 0:
        nemo_conv_input = nemo_conv_inputs[0]
        if isinstance(nemo_conv_input, (list, tuple)) and len(nemo_conv_input) > 0:
            nemo_conv_input_tensor = nemo_conv_input[0]
        else:
            nemo_conv_input_tensor = nemo_conv_input
        
        if isinstance(nemo_conv_input_tensor, torch.Tensor):
            print(f"  NeMo: shape={nemo_conv_input_tensor.shape}")
            print(f"  NeMo: mean={nemo_conv_input_tensor.float().mean():.6f}")
            
            # Compare with transposed
            if nemo_conv_input_tensor.shape == nemo_transposed.shape:
                diff = (nemo_conv_input_tensor.float() - nemo_transposed.float()).abs()
                max_diff = diff.max().item()
                print(f"  NeMo vs transposed: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
            elif nemo_conv_input_tensor.shape == nemo_transposed.unsqueeze(1).shape:
                print(f"  [NOTE] NeMo hook captured unsqueezed input (shape matches unsqueeze(1))")
                # Compare with unsqueezed
                nemo_unsqueezed = nemo_transposed.unsqueeze(1)
                diff = (nemo_conv_input_tensor.float() - nemo_unsqueezed.float()).abs()
                max_diff = diff.max().item()
                print(f"  NeMo vs unsqueezed: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
    
    if len(nest_conv_inputs) > 0:
        nest_conv_input = nest_conv_inputs[0]
        if isinstance(nest_conv_input, (list, tuple)) and len(nest_conv_input) > 0:
            nest_conv_input_tensor = nest_conv_input[0]
        else:
            nest_conv_input_tensor = nest_conv_input
        
        if isinstance(nest_conv_input_tensor, torch.Tensor):
            print(f"  nest: shape={nest_conv_input_tensor.shape}")
            print(f"  nest: mean={nest_conv_input_tensor.float().mean():.6f}")
            
            # Compare with transposed
            if nest_conv_input_tensor.shape == nest_transposed.shape:
                diff = (nest_conv_input_tensor.float() - nest_transposed.float()).abs()
                max_diff = diff.max().item()
                print(f"  nest vs transposed: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
            elif nest_conv_input_tensor.shape == nest_transposed.unsqueeze(1).shape:
                print(f"  [NOTE] nest hook captured unsqueezed input (shape matches unsqueeze(1))")
                # Compare with unsqueezed
                nest_unsqueezed = nest_transposed.unsqueeze(1)
                diff = (nest_conv_input_tensor.float() - nest_unsqueezed.float()).abs()
                max_diff = diff.max().item()
                print(f"  nest vs unsqueezed: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")


if __name__ == '__main__':
    main()

