#!/usr/bin/env python3
"""
Check when conv hook captures data - before or after mask application.
"""

import torch
import pickle
from pathlib import Path


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK CONV HOOK TIMING")
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
    
    nemo_seq_len = nemo_preproc_outs[1][1]  # Call 1, output[1] (seq_len)
    nest_seq_len = nest_preproc_outs[1][1]  # Call 1, output[1] (seq_len)
    
    # Expected MaskedConvSequential input: [B, T, D] after transpose
    nemo_expected_conv_input = nemo_preproc_call1.transpose(1, 2)  # [B, T, D]
    nest_expected_conv_input = nest_preproc_call1.transpose(1, 2)  # [B, T, D]
    
    print(f"\nExpected MaskedConvSequential input (after transpose):")
    print(f"  NeMo: shape={nemo_expected_conv_input.shape}, mean={nemo_expected_conv_input.float().mean():.6f}")
    print(f"  nest: shape={nest_expected_conv_input.shape}, mean={nest_expected_conv_input.float().mean():.6f}")
    
    # Get actual MaskedConvSequential input (from encoder.pre_encode.conv hook)
    nemo_conv = nemo_layers.get('encoder.pre_encode.conv', {})
    nest_conv = nest_layers.get('encoder.pre_encode.conv', {})
    
    nemo_conv_inputs = nemo_conv.get('all_forward_inputs', [])
    nest_conv_inputs = nest_conv.get('all_forward_inputs', [])
    
    print(f"\nActual MaskedConvSequential input (from encoder.pre_encode.conv hook):")
    if len(nemo_conv_inputs) > 0:
        nemo_conv_input = nemo_conv_inputs[0]
        if isinstance(nemo_conv_input, (list, tuple)) and len(nemo_conv_input) > 0:
            nemo_conv_input_tensor = nemo_conv_input[0]
        else:
            nemo_conv_input_tensor = nemo_conv_input
        
        if isinstance(nemo_conv_input_tensor, torch.Tensor):
            print(f"  NeMo: shape={nemo_conv_input_tensor.shape}, mean={nemo_conv_input_tensor.float().mean():.6f}")
            
            # Compare with expected
            if nemo_conv_input_tensor.shape == nemo_expected_conv_input.shape:
                diff = (nemo_conv_input_tensor.float() - nemo_expected_conv_input.float()).abs()
                max_diff = diff.max().item()
                print(f"    vs expected: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
            elif nemo_conv_input_tensor.shape == nemo_expected_conv_input.unsqueeze(1).shape:
                nemo_expected_unsqueezed = nemo_expected_conv_input.unsqueeze(1)
                diff = (nemo_conv_input_tensor.float() - nemo_expected_unsqueezed.float()).abs()
                max_diff = diff.max().item()
                print(f"    vs expected (unsqueezed): max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
    
    if len(nest_conv_inputs) > 0:
        nest_conv_input = nest_conv_inputs[0]
        if isinstance(nest_conv_input, (list, tuple)) and len(nest_conv_input) > 0:
            nest_conv_input_tensor = nest_conv_input[0]
        else:
            nest_conv_input_tensor = nest_conv_input
        
        if isinstance(nest_conv_input_tensor, torch.Tensor):
            print(f"  nest: shape={nest_conv_input_tensor.shape}, mean={nest_conv_input_tensor.float().mean():.6f}")
            
            # Compare with expected
            if nest_conv_input_tensor.shape == nest_expected_conv_input.shape:
                diff = (nest_conv_input_tensor.float() - nest_expected_conv_input.float()).abs()
                max_diff = diff.max().item()
                print(f"    vs expected: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
            elif nest_conv_input_tensor.shape == nest_expected_conv_input.unsqueeze(1).shape:
                nest_expected_unsqueezed = nest_expected_conv_input.unsqueeze(1)
                diff = (nest_conv_input_tensor.float() - nest_expected_unsqueezed.float()).abs()
                max_diff = diff.max().item()
                print(f"    vs expected (unsqueezed): max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
    
    # Get conv.0 input
    nemo_conv0 = nemo_layers.get('encoder.pre_encode.conv.0', {})
    nest_conv0 = nest_layers.get('encoder.pre_encode.conv.0', {})
    
    nemo_conv0_inputs = nemo_conv0.get('all_forward_inputs', [])
    nest_conv0_inputs = nest_conv0.get('all_forward_inputs', [])
    
    print(f"\nActual conv.0 input (from encoder.pre_encode.conv.0 hook):")
    if len(nemo_conv0_inputs) > 0:
        nemo_conv0_input = nemo_conv0_inputs[0]
        if isinstance(nemo_conv0_input, (list, tuple)) and len(nemo_conv0_input) > 0:
            nemo_conv0_input_tensor = nemo_conv0_input[0]
        else:
            nemo_conv0_input_tensor = nemo_conv0_input
        
        if isinstance(nemo_conv0_input_tensor, torch.Tensor):
            print(f"  NeMo: shape={nemo_conv0_input_tensor.shape}, mean={nemo_conv0_input_tensor.float().mean():.6f}")
    
    if len(nest_conv0_inputs) > 0:
        nest_conv0_input = nest_conv0_inputs[0]
        if isinstance(nest_conv0_input, (list, tuple)) and len(nest_conv0_input) > 0:
            nest_conv0_input_tensor = nest_conv0_input[0]
        else:
            nest_conv0_input_tensor = nest_conv0_input
        
        if isinstance(nest_conv0_input_tensor, torch.Tensor):
            print(f"  nest: shape={nest_conv0_input_tensor.shape}, mean={nest_conv0_input_tensor.float().mean():.6f}")


if __name__ == '__main__':
    main()

