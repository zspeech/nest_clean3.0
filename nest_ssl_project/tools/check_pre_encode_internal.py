#!/usr/bin/env python3
"""
Check encoder.pre_encode.pre_encode (MaskedConvSequential) output.
"""

import torch
import pickle
from pathlib import Path

def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK PRE_ENCODE INTERNAL OUTPUT")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # In NeMo with Wrapper, the inner module is encoder.pre_encode.pre_encode
    # In nest with Wrapper, it should be the same
    
    # Check keys
    print("Searching for pre_encode keys...")
    nemo_keys = [k for k in nemo_layers.keys() if 'pre_encode' in k]
    nest_keys = [k for k in nest_layers.keys() if 'pre_encode' in k]
    
    print(f"NeMo keys: {nemo_keys[:5]}...")
    print(f"nest keys: {nest_keys[:5]}...")
    
    # Try to find the inner ConvSubsampling output
    # The wrapper is 'encoder.pre_encode'
    # The inner ConvSubsampling is 'encoder.pre_encode.pre_encode' (if wrapper stores it as .pre_encode)
    
    target_key = 'encoder.pre_encode.pre_encode'
    
    nemo_out = None
    nest_out = None
    
    if target_key in nemo_layers:
        outs = nemo_layers[target_key].get('all_forward_outputs', [])
        if len(outs) > 0:
            nemo_out = outs[-1][0] # Last call
            print(f"NeMo {target_key} output shape: {nemo_out.shape}, mean: {nemo_out.float().mean():.6f}")
            
    if target_key in nest_layers:
        outs = nest_layers[target_key].get('all_forward_outputs', [])
        if len(outs) > 0:
            nest_out = outs[-1][0] # Last call
            print(f"nest {target_key} output shape: {nest_out.shape}, mean: {nest_out.float().mean():.6f}")

    if nemo_out is not None and nest_out is not None:
        if nemo_out.shape == nest_out.shape:
            diff = (nemo_out.float() - nest_out.float()).abs().max().item()
            print(f"NeMo vs nest {target_key} max diff: {diff:.6e} {'[OK]' if diff < 1e-5 else '[FAIL]'}")

if __name__ == '__main__':
    main()

