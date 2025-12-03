#!/usr/bin/env python3
"""
Simple trace of encoder input without complex imports.
"""

import torch
import pickle
from pathlib import Path
import sys


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("SIMPLE ENCODER INPUT TRACE")
    print("="*80)
    
    # Load layer outputs
    print("\nLoading NeMo layer outputs...")
    try:
        with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
            nemo_layers = pickle.load(f)
        print(f"  Loaded {len(nemo_layers)} layers")
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    
    print("\nLoading nest layer outputs...")
    try:
        with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
            nest_layers = pickle.load(f)
        print(f"  Loaded {len(nest_layers)} layers")
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    
    # Check preprocessor
    print("\n" + "="*80)
    print("PREPROCESSOR OUTPUTS")
    print("="*80)
    
    if 'preprocessor' in nemo_layers:
        nemo_preproc = nemo_layers['preprocessor']
        all_outs = nemo_preproc.get('all_forward_outputs', [])
        print(f"\nNeMo preprocessor calls: {len(all_outs)}")
        for i, out in enumerate(all_outs):
            if isinstance(out, (list, tuple)) and len(out) > 0:
                if isinstance(out[0], torch.Tensor):
                    print(f"  Call {i}: shape={out[0].shape}, mean={out[0].float().mean():.6f}")
    
    if 'preprocessor' in nest_layers:
        nest_preproc = nest_layers['preprocessor']
        all_outs = nest_preproc.get('all_forward_outputs', [])
        print(f"\nnest preprocessor calls: {len(all_outs)}")
        for i, out in enumerate(all_outs):
            if isinstance(out, (list, tuple)) and len(out) > 0:
                if isinstance(out[0], torch.Tensor):
                    print(f"  Call {i}: shape={out[0].shape}, mean={out[0].float().mean():.6f}")
    
    # Check encoder
    print("\n" + "="*80)
    print("ENCODER INPUTS")
    print("="*80)
    
    if 'encoder' in nemo_layers:
        nemo_encoder = nemo_layers['encoder']
        all_ins = nemo_encoder.get('all_forward_inputs', [])
        print(f"\nNeMo encoder calls: {len(all_ins)}")
        for i, inp in enumerate(all_ins):
            if isinstance(inp, (list, tuple)) and len(inp) > 0:
                if isinstance(inp[0], torch.Tensor):
                    print(f"  Call {i}: shape={inp[0].shape}, mean={inp[0].float().mean():.6f}")
    
    if 'encoder' in nest_layers:
        nest_encoder = nest_layers['encoder']
        all_ins = nest_encoder.get('all_forward_inputs', [])
        print(f"\nnest encoder calls: {len(all_ins)}")
        for i, inp in enumerate(all_ins):
            if isinstance(inp, (list, tuple)) and len(inp) > 0:
                if isinstance(inp[0], torch.Tensor):
                    print(f"  Call {i}: shape={inp[0].shape}, mean={inp[0].float().mean():.6f}")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    nemo_preproc_outs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preproc_outs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nemo_encoder_ins = nemo_layers.get('encoder', {}).get('all_forward_inputs', [])
    nest_encoder_ins = nest_layers.get('encoder', {}).get('all_forward_inputs', [])
    
    # Compare preprocessor Call 1 outputs
    if len(nemo_preproc_outs) > 1 and len(nest_preproc_outs) > 1:
        nemo_p1 = nemo_preproc_outs[1][0]
        nest_p1 = nest_preproc_outs[1][0]
        diff = (nemo_p1 - nest_p1).abs().max().item()
        print(f"\nPreprocessor Call 1 output: max_diff={diff:.6e}")
    
    # Compare encoder inputs
    if len(nemo_encoder_ins) > 0 and len(nest_encoder_ins) > 0:
        nemo_e = nemo_encoder_ins[0][0]
        nest_e = nest_encoder_ins[0][0]
        diff = (nemo_e - nest_e).abs().max().item()
        print(f"Encoder input: max_diff={diff:.6e}")
    
    # Compare encoder input vs preprocessor Call 1 output
    if len(nemo_encoder_ins) > 0 and len(nemo_preproc_outs) > 1:
        nemo_e = nemo_encoder_ins[0][0]
        nemo_p1 = nemo_preproc_outs[1][0]
        diff = (nemo_e - nemo_p1).abs().max().item()
        print(f"\nNeMo: encoder_input vs preprocessor_Call1: max_diff={diff:.6e}")
    
    if len(nest_encoder_ins) > 0 and len(nest_preproc_outs) > 1:
        nest_e = nest_encoder_ins[0][0]
        nest_p1 = nest_preproc_outs[1][0]
        diff = (nest_e - nest_p1).abs().max().item()
        print(f"nest: encoder_input vs preprocessor_Call1: max_diff={diff:.6e}")


if __name__ == '__main__':
    main()

