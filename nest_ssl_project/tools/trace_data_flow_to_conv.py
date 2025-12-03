#!/usr/bin/env python3
"""
Trace data flow from preprocessor output to MaskedConvSequential input.
"""

import torch
import pickle
from pathlib import Path


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("TRACE DATA FLOW TO CONV")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Step 1: Preprocessor Call 1 output
    nemo_preproc_outs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preproc_outs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    
    nemo_preproc_call1 = nemo_preproc_outs[1][0]  # Call 1, output[0]
    nest_preproc_call1 = nest_preproc_outs[1][0]  # Call 1, output[0]
    
    print(f"\nStep 1: Preprocessor Call 1 output")
    print(f"  NeMo: shape={nemo_preproc_call1.shape}, mean={nemo_preproc_call1.float().mean():.6f}")
    print(f"  nest: shape={nest_preproc_call1.shape}, mean={nest_preproc_call1.float().mean():.6f}")
    
    # Step 2: After transpose(1, 2) in ConformerEncoder.forward_internal
    nemo_transposed = nemo_preproc_call1.transpose(1, 2)
    nest_transposed = nest_preproc_call1.transpose(1, 2)
    
    print(f"\nStep 2: After transpose(1, 2) in ConformerEncoder.forward_internal")
    print(f"  NeMo: shape={nemo_transposed.shape}, mean={nemo_transposed.float().mean():.6f}")
    print(f"  nest: shape={nest_transposed.shape}, mean={nest_transposed.float().mean():.6f}")
    
    # Step 3: Actual MaskedConvSequential input (from hook)
    nemo_conv = nemo_layers.get('encoder.pre_encode.conv', {})
    nest_conv = nest_layers.get('encoder.pre_encode.conv', {})
    
    nemo_conv_inputs = nemo_conv.get('all_forward_inputs', [])
    nest_conv_inputs = nest_conv.get('all_forward_inputs', [])
    
    if len(nemo_conv_inputs) > 0:
        nemo_conv_input = nemo_conv_inputs[0]
        if isinstance(nemo_conv_input, (list, tuple)) and len(nemo_conv_input) > 0:
            nemo_conv_input_tensor = nemo_conv_input[0]
        else:
            nemo_conv_input_tensor = nemo_conv_input
        
        if isinstance(nemo_conv_input_tensor, torch.Tensor):
            print(f"\nStep 3: Actual MaskedConvSequential input (from hook)")
            print(f"  NeMo: shape={nemo_conv_input_tensor.shape}, mean={nemo_conv_input_tensor.float().mean():.6f}")
            
            # Compare with Step 2
            if nemo_conv_input_tensor.shape == nemo_transposed.shape:
                diff = (nemo_conv_input_tensor.float() - nemo_transposed.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"  NeMo Step 3 vs Step 2: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
            elif nemo_conv_input_tensor.shape == nemo_transposed.unsqueeze(1).shape:
                nemo_unsqueezed = nemo_transposed.unsqueeze(1)
                diff = (nemo_conv_input_tensor.float() - nemo_unsqueezed.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"  NeMo Step 3 vs Step 2 (unsqueezed): max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
    
    if len(nest_conv_inputs) > 0:
        nest_conv_input = nest_conv_inputs[0]
        if isinstance(nest_conv_input, (list, tuple)) and len(nest_conv_input) > 0:
            nest_conv_input_tensor = nest_conv_input[0]
        else:
            nest_conv_input_tensor = nest_conv_input
        
        if isinstance(nest_conv_input_tensor, torch.Tensor):
            print(f"  nest: shape={nest_conv_input_tensor.shape}, mean={nest_conv_input_tensor.float().mean():.6f}")
            
            # Compare with Step 2
            if nest_conv_input_tensor.shape == nest_transposed.shape:
                diff = (nest_conv_input_tensor.float() - nest_transposed.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"  nest Step 3 vs Step 2: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
            elif nest_conv_input_tensor.shape == nest_transposed.unsqueeze(1).shape:
                nest_unsqueezed = nest_transposed.unsqueeze(1)
                diff = (nest_conv_input_tensor.float() - nest_unsqueezed.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"  nest Step 3 vs Step 2 (unsqueezed): max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
    
    # Step 4: conv.0 input (after unsqueeze and mask in MaskedConvSequential)
    nemo_conv0 = nemo_layers.get('encoder.pre_encode.conv.0', {})
    nest_conv0 = nest_layers.get('encoder.pre_encode.conv.0', {})
    
    nemo_conv0_inputs = nemo_conv0.get('all_forward_inputs', [])
    nest_conv0_inputs = nest_conv0.get('all_forward_inputs', [])
    
    if len(nemo_conv0_inputs) > 0 and len(nest_conv0_inputs) > 0:
        nemo_conv0_input = nemo_conv0_inputs[0]
        nest_conv0_input = nest_conv0_inputs[0]
        
        if isinstance(nemo_conv0_input, (list, tuple)) and len(nemo_conv0_input) > 0:
            nemo_conv0_input_tensor = nemo_conv0_input[0]
        else:
            nemo_conv0_input_tensor = nemo_conv0_input
        
        if isinstance(nest_conv0_input, (list, tuple)) and len(nest_conv0_input) > 0:
            nest_conv0_input_tensor = nest_conv0_input[0]
        else:
            nest_conv0_input_tensor = nest_conv0_input
        
        if isinstance(nemo_conv0_input_tensor, torch.Tensor) and isinstance(nest_conv0_input_tensor, torch.Tensor):
            print(f"\nStep 4: conv.0 input (after unsqueeze and mask)")
            print(f"  NeMo: shape={nemo_conv0_input_tensor.shape}, mean={nemo_conv0_input_tensor.float().mean():.6f}")
            print(f"  nest: shape={nest_conv0_input_tensor.shape}, mean={nest_conv0_input_tensor.float().mean():.6f}")
            
            # Compare NeMo vs nest
            if nemo_conv0_input_tensor.shape == nest_conv0_input_tensor.shape:
                diff = (nemo_conv0_input_tensor.float() - nest_conv0_input_tensor.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"  NeMo vs nest: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")


if __name__ == '__main__':
    main()

