#!/usr/bin/env python3
"""
Check all hook locations related to pre_encode.
"""

import torch
import pickle
from pathlib import Path


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK HOOK LOCATIONS (pre_encode related)")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Find all pre_encode related keys
    print("\nNeMo pre_encode related keys:")
    nemo_pre_encode_keys = [k for k in nemo_layers.keys() if 'pre_encode' in k]
    for key in sorted(nemo_pre_encode_keys):
        print(f"  {key}")
    
    print("\nnest pre_encode related keys:")
    nest_pre_encode_keys = [k for k in nest_layers.keys() if 'pre_encode' in k]
    for key in sorted(nest_pre_encode_keys):
        print(f"  {key}")
    
    # Check if encoder.pre_encode.pre_encode.conv exists
    print("\n" + "="*80)
    print("CHECK encoder.pre_encode.pre_encode.conv (if ConvFeatureMaskingWrapper is used)")
    print("="*80)
    
    nemo_pre_encode_pre_encode_conv = nemo_layers.get('encoder.pre_encode.pre_encode.conv', {})
    nest_pre_encode_pre_encode_conv = nest_layers.get('encoder.pre_encode.pre_encode.conv', {})
    
    print(f"\nNeMo encoder.pre_encode.pre_encode.conv: {'FOUND' if nemo_pre_encode_pre_encode_conv else 'NOT FOUND'}")
    print(f"nest encoder.pre_encode.pre_encode.conv: {'FOUND' if nest_pre_encode_pre_encode_conv else 'NOT FOUND'}")
    
    # Check encoder.pre_encode.conv
    print("\n" + "="*80)
    print("CHECK encoder.pre_encode.conv")
    print("="*80)
    
    nemo_pre_encode_conv = nemo_layers.get('encoder.pre_encode.conv', {})
    nest_pre_encode_conv = nest_layers.get('encoder.pre_encode.conv', {})
    
    print(f"\nNeMo encoder.pre_encode.conv: {'FOUND' if nemo_pre_encode_conv else 'NOT FOUND'}")
    print(f"nest encoder.pre_encode.conv: {'FOUND' if nest_pre_encode_conv else 'NOT FOUND'}")
    
    # Check if mask_position is post_conv (which would replace encoder.pre_encode)
    print("\n" + "="*80)
    print("CHECK encoder.pre_encode type")
    print("="*80)
    
    nemo_pre_encode = nemo_layers.get('encoder.pre_encode', {})
    nest_pre_encode = nest_layers.get('encoder.pre_encode', {})
    
    if nemo_pre_encode:
        nemo_pre_encode_inputs = nemo_pre_encode.get('all_forward_inputs', [])
        if len(nemo_pre_encode_inputs) > 0:
            first_input = nemo_pre_encode_inputs[0]
            if isinstance(first_input, (list, tuple)) and len(first_input) > 0:
                first_input_tensor = first_input[0]
            else:
                first_input_tensor = first_input
            if isinstance(first_input_tensor, torch.Tensor):
                print(f"\nNeMo encoder.pre_encode input shape: {first_input_tensor.shape}")
    
    if nest_pre_encode:
        nest_pre_encode_inputs = nest_pre_encode.get('all_forward_inputs', [])
        if len(nest_pre_encode_inputs) > 0:
            first_input = nest_pre_encode_inputs[0]
            if isinstance(first_input, (list, tuple)) and len(first_input) > 0:
                first_input_tensor = first_input[0]
            else:
                first_input_tensor = first_input
            if isinstance(first_input_tensor, torch.Tensor):
                print(f"nest encoder.pre_encode input shape: {first_input_tensor.shape}")


if __name__ == '__main__':
    main()

