#!/usr/bin/env python3
"""
Check which conv-related hooks are registered and what they capture.
"""

import torch
import pickle
from pathlib import Path


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK CONV HOOK LOCATIONS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Find all conv-related keys
    print("\nNeMo conv-related keys:")
    nemo_conv_keys = [k for k in nemo_layers.keys() if 'conv' in k.lower() and 'pre_encode' in k]
    for key in sorted(nemo_conv_keys):
        print(f"  {key}")
        hook_data = nemo_layers[key]
        all_inputs = hook_data.get('all_forward_inputs', [])
        if len(all_inputs) > 0:
            first_input = all_inputs[0]
            if isinstance(first_input, (list, tuple)) and len(first_input) > 0:
                first_input_tensor = first_input[0]
            else:
                first_input_tensor = first_input
            if isinstance(first_input_tensor, torch.Tensor):
                print(f"    Input shape: {first_input_tensor.shape}")
    
    print("\nnest conv-related keys:")
    nest_conv_keys = [k for k in nest_layers.keys() if 'conv' in k.lower() and 'pre_encode' in k]
    for key in sorted(nest_conv_keys):
        print(f"  {key}")
        hook_data = nest_layers[key]
        all_inputs = hook_data.get('all_forward_inputs', [])
        if len(all_inputs) > 0:
            first_input = all_inputs[0]
            if isinstance(first_input, (list, tuple)) and len(first_input) > 0:
                first_input_tensor = first_input[0]
            else:
                first_input_tensor = first_input
            if isinstance(first_input_tensor, torch.Tensor):
                print(f"    Input shape: {first_input_tensor.shape}")
    
    # Compare encoder.pre_encode.conv vs encoder.pre_encode.conv.0
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    nemo_conv = nemo_layers.get('encoder.pre_encode.conv', {})
    nest_conv = nest_layers.get('encoder.pre_encode.conv', {})
    
    nemo_conv0 = nemo_layers.get('encoder.pre_encode.conv.0', {})
    nest_conv0 = nest_layers.get('encoder.pre_encode.conv.0', {})
    
    print("\nencoder.pre_encode.conv:")
    if nemo_conv:
        nemo_inputs = nemo_conv.get('all_forward_inputs', [])
        if len(nemo_inputs) > 0:
            nemo_input = nemo_inputs[0]
            if isinstance(nemo_input, (list, tuple)) and len(nemo_input) > 0:
                nemo_input_tensor = nemo_input[0]
            else:
                nemo_input_tensor = nemo_input
            if isinstance(nemo_input_tensor, torch.Tensor):
                print(f"  NeMo: shape={nemo_input_tensor.shape}")
    
    if nest_conv:
        nest_inputs = nest_conv.get('all_forward_inputs', [])
        if len(nest_inputs) > 0:
            nest_input = nest_inputs[0]
            if isinstance(nest_input, (list, tuple)) and len(nest_input) > 0:
                nest_input_tensor = nest_input[0]
            else:
                nest_input_tensor = nest_input
            if isinstance(nest_input_tensor, torch.Tensor):
                print(f"  nest: shape={nest_input_tensor.shape}")
    
    print("\nencoder.pre_encode.conv.0:")
    if nemo_conv0:
        nemo_inputs0 = nemo_conv0.get('all_forward_inputs', [])
        if len(nemo_inputs0) > 0:
            nemo_input0 = nemo_inputs0[0]
            if isinstance(nemo_input0, (list, tuple)) and len(nemo_input0) > 0:
                nemo_input0_tensor = nemo_input0[0]
            else:
                nemo_input0_tensor = nemo_input0
            if isinstance(nemo_input0_tensor, torch.Tensor):
                print(f"  NeMo: shape={nemo_input0_tensor.shape}")
    
    if nest_conv0:
        nest_inputs0 = nest_conv0.get('all_forward_inputs', [])
        if len(nest_inputs0) > 0:
            nest_input0 = nest_inputs0[0]
            if isinstance(nest_input0, (list, tuple)) and len(nest_input0) > 0:
                nest_input0_tensor = nest_input0[0]
            else:
                nest_input0_tensor = nest_input0
            if isinstance(nest_input0_tensor, torch.Tensor):
                print(f"  nest: shape={nest_input0_tensor.shape}")


if __name__ == '__main__':
    main()

