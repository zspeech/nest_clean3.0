#!/usr/bin/env python3
"""
Check encoder.pre_encode input vs preprocessor outputs.
"""

import torch
import pickle
from pathlib import Path


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK PRE_ENCODE INPUT VS PREPROCESSOR OUTPUTS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get preprocessor outputs
    nemo_preproc_outs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preproc_outs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    
    print(f"\nPreprocessor calls:")
    print(f"  NeMo: {len(nemo_preproc_outs)} calls")
    print(f"  nest: {len(nest_preproc_outs)} calls")
    
    for i, (nemo_out, nest_out) in enumerate(zip(nemo_preproc_outs, nest_preproc_outs)):
        nemo_out_tensor = nemo_out[0]
        nest_out_tensor = nest_out[0]
        print(f"\n  Call {i}:")
        print(f"    NeMo: shape={nemo_out_tensor.shape}, mean={nemo_out_tensor.float().mean():.6f}")
        print(f"    nest: shape={nest_out_tensor.shape}, mean={nest_out_tensor.float().mean():.6f}")
    
    # Get pre_encode inputs
    nemo_pre_encode = nemo_layers.get('encoder.pre_encode', {})
    nest_pre_encode = nest_layers.get('encoder.pre_encode', {})
    
    nemo_pre_encode_inputs = nemo_pre_encode.get('all_forward_inputs', [])
    nest_pre_encode_inputs = nest_pre_encode.get('all_forward_inputs', [])
    
    print(f"\npre_encode inputs:")
    print(f"  NeMo: {len(nemo_pre_encode_inputs)} calls")
    print(f"  nest: {len(nest_pre_encode_inputs)} calls")
    
    if len(nemo_pre_encode_inputs) > 0:
        nemo_input = nemo_pre_encode_inputs[0]
        if isinstance(nemo_input, (list, tuple)) and len(nemo_input) > 0:
            nemo_input_tensor = nemo_input[0]
        else:
            nemo_input_tensor = nemo_input
        
        if isinstance(nemo_input_tensor, torch.Tensor):
            print(f"\n  NeMo pre_encode input:")
            print(f"    shape={nemo_input_tensor.shape}, mean={nemo_input_tensor.float().mean():.6f}")
            
            # Compare with preprocessor outputs
            for i, nemo_out in enumerate(nemo_preproc_outs):
                nemo_out_tensor = nemo_out[0].transpose(1, 2)  # transpose to [B, T, D]
                if nemo_input_tensor.shape == nemo_out_tensor.shape:
                    diff = (nemo_input_tensor.float() - nemo_out_tensor.float()).abs()
                    max_diff = diff.max().item()
                    print(f"    vs preprocessor Call {i} (transposed): max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
    
    if len(nest_pre_encode_inputs) > 0:
        nest_input = nest_pre_encode_inputs[0]
        if isinstance(nest_input, (list, tuple)) and len(nest_input) > 0:
            nest_input_tensor = nest_input[0]
        else:
            nest_input_tensor = nest_input
        
        if isinstance(nest_input_tensor, torch.Tensor):
            print(f"\n  nest pre_encode input:")
            print(f"    shape={nest_input_tensor.shape}, mean={nest_input_tensor.float().mean():.6f}")
            
            # Compare with preprocessor outputs
            for i, nest_out in enumerate(nest_preproc_outs):
                nest_out_tensor = nest_out[0].transpose(1, 2)  # transpose to [B, T, D]
                if nest_input_tensor.shape == nest_out_tensor.shape:
                    diff = (nest_input_tensor.float() - nest_out_tensor.float()).abs()
                    max_diff = diff.max().item()
                    print(f"    vs preprocessor Call {i} (transposed): max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")


if __name__ == '__main__':
    main()

