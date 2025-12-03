#!/usr/bin/env python3
"""
Trace data flow from preprocessor output to pre_encode.conv input
"""

import torch
import pickle
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_dir", type=str, required=True)
    parser.add_argument("--nest_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir)
    nest_dir = Path(args.nest_dir)
    
    if args.step is None:
        nemo_steps = [int(d.name.split('_')[1]) for d in nemo_dir.glob("step_*") if d.is_dir()]
        nest_steps = [int(d.name.split('_')[1]) for d in nest_dir.glob("step_*") if d.is_dir()]
        common_steps = sorted(list(set(nemo_steps) & set(nest_steps)))
        if common_steps:
            args.step = common_steps[0]
    
    print("="*80)
    print("DATA FLOW TRACE")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get preprocessor output
    nemo_prep_out = nemo_layers.get('preprocessor', {}).get('forward_outputs')
    nest_prep_out = nest_layers.get('preprocessor', {}).get('forward_outputs')
    
    if nemo_prep_out and nest_prep_out:
        nemo_prep = nemo_prep_out[0] if isinstance(nemo_prep_out, (list, tuple)) else nemo_prep_out
        nest_prep = nest_prep_out[0] if isinstance(nest_prep_out, (list, tuple)) else nest_prep_out
        
        print("\n" + "="*60)
        print("1. Preprocessor Output")
        print("="*60)
        print(f"NeMo: shape={nemo_prep.shape}, mean={nemo_prep.mean().item():.6f}")
        print(f"nest: shape={nest_prep.shape}, mean={nest_prep.mean().item():.6f}")
        print(f"Diff: max={( nemo_prep - nest_prep).abs().max().item():.6e}")
        
        # Simulate the transformation: transpose then unsqueeze
        print("\n" + "="*60)
        print("2. After transpose(1, 2)")
        print("="*60)
        nemo_transposed = nemo_prep.transpose(1, 2)
        nest_transposed = nest_prep.transpose(1, 2)
        print(f"NeMo: shape={nemo_transposed.shape}, mean={nemo_transposed.mean().item():.6f}")
        print(f"nest: shape={nest_transposed.shape}, mean={nest_transposed.mean().item():.6f}")
        print(f"Diff: max={(nemo_transposed - nest_transposed).abs().max().item():.6e}")
        
        print("\n" + "="*60)
        print("3. After unsqueeze(1)")
        print("="*60)
        nemo_unsqueezed = nemo_transposed.unsqueeze(1)
        nest_unsqueezed = nest_transposed.unsqueeze(1)
        print(f"NeMo: shape={nemo_unsqueezed.shape}, mean={nemo_unsqueezed.mean().item():.6f}")
        print(f"nest: shape={nest_unsqueezed.shape}, mean={nest_unsqueezed.mean().item():.6f}")
        print(f"Diff: max={(nemo_unsqueezed - nest_unsqueezed).abs().max().item():.6e}")
    
    # Get actual conv input
    nemo_conv_in = nemo_layers.get('encoder.pre_encode.conv', {}).get('forward_inputs')
    nest_conv_in = nest_layers.get('encoder.pre_encode.conv', {}).get('forward_inputs')
    
    if nemo_conv_in and nest_conv_in:
        nemo_conv = nemo_conv_in[0] if isinstance(nemo_conv_in, (list, tuple)) else nemo_conv_in
        nest_conv = nest_conv_in[0] if isinstance(nest_conv_in, (list, tuple)) else nest_conv_in
        
        print("\n" + "="*60)
        print("4. Actual pre_encode.conv Input (from hooks)")
        print("="*60)
        print(f"NeMo: shape={nemo_conv.shape}, mean={nemo_conv.mean().item():.6f}")
        print(f"nest: shape={nest_conv.shape}, mean={nest_conv.mean().item():.6f}")
        print(f"Diff: max={(nemo_conv - nest_conv).abs().max().item():.6e}")
        
        # Compare simulated vs actual
        print("\n" + "="*60)
        print("5. Simulated vs Actual")
        print("="*60)
        print(f"NeMo simulated vs actual: max_diff={(nemo_unsqueezed - nemo_conv).abs().max().item():.6e}")
        print(f"nest simulated vs actual: max_diff={(nest_unsqueezed - nest_conv).abs().max().item():.6e}")
        
        # Check if NeMo's conv input matches nest's simulated
        print("\n" + "="*60)
        print("6. Cross comparison")
        print("="*60)
        print(f"NeMo actual vs nest simulated: max_diff={(nemo_conv - nest_unsqueezed).abs().max().item():.6e}")
        print(f"nest actual vs nemo simulated: max_diff={(nest_conv - nemo_unsqueezed).abs().max().item():.6e}")
        
        # Check specific values
        print("\n" + "="*60)
        print("7. Sample values at position [0, 0, 0, 0]")
        print("="*60)
        print(f"NeMo preprocessor: {nemo_prep[0, 0, 0].item():.6f}")
        print(f"nest preprocessor: {nest_prep[0, 0, 0].item():.6f}")
        print(f"NeMo conv input: {nemo_conv[0, 0, 0, 0].item():.6f}")
        print(f"nest conv input: {nest_conv[0, 0, 0, 0].item():.6f}")
        print(f"NeMo simulated: {nemo_unsqueezed[0, 0, 0, 0].item():.6f}")
        print(f"nest simulated: {nest_unsqueezed[0, 0, 0, 0].item():.6f}")
        
        # Check at max diff position
        diff = (nemo_conv - nest_conv).abs()
        max_idx = diff.argmax()
        idx = []
        flat_idx = max_idx.item()
        for dim in reversed(nemo_conv.shape):
            idx.insert(0, flat_idx % dim)
            flat_idx //= dim
        idx = tuple(idx)
        
        print(f"\n  At max diff position {idx}:")
        print(f"    NeMo conv input: {nemo_conv[idx].item():.6f}")
        print(f"    nest conv input: {nest_conv[idx].item():.6f}")
        print(f"    NeMo simulated: {nemo_unsqueezed[idx].item():.6f}")
        print(f"    nest simulated: {nest_unsqueezed[idx].item():.6f}")
        
        # The corresponding position in preprocessor output
        # conv input [B, 1, T, F] -> preprocessor [B, F, T]
        b, _, t, f = idx
        prep_idx = (b, f, t)
        print(f"\n  Corresponding preprocessor position {prep_idx}:")
        print(f"    NeMo preprocessor: {nemo_prep[prep_idx].item():.6f}")
        print(f"    nest preprocessor: {nest_prep[prep_idx].item():.6f}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

