#!/usr/bin/env python3
"""
Diagnose pre_encode input - check what happens between preprocessor output and pre_encode.conv input
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
    print("PRE_ENCODE INPUT DIAGNOSIS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # 1. Check preprocessor output (should be [OK])
    print("\n" + "="*60)
    print("1. Preprocessor Output")
    print("="*60)
    
    nemo_prep = nemo_layers.get('preprocessor', {}).get('forward_outputs')
    nest_prep = nest_layers.get('preprocessor', {}).get('forward_outputs')
    
    if nemo_prep and nest_prep:
        nemo_out = nemo_prep[0] if isinstance(nemo_prep, (list, tuple)) else nemo_prep
        nest_out = nest_prep[0] if isinstance(nest_prep, (list, tuple)) else nest_prep
        
        print(f"NeMo shape: {nemo_out.shape}")
        print(f"nest shape: {nest_out.shape}")
        
        diff = (nemo_out - nest_out).abs()
        print(f"Max diff: {diff.max().item():.6e}")
        print(f"Mean diff: {diff.mean().item():.6e}")
    
    # 2. Check encoder input (from forward_inputs)
    print("\n" + "="*60)
    print("2. Encoder Forward Inputs")
    print("="*60)
    
    nemo_enc_inputs = nemo_layers.get('encoder', {}).get('forward_inputs')
    nest_enc_inputs = nest_layers.get('encoder', {}).get('forward_inputs')
    
    if nemo_enc_inputs:
        print(f"NeMo encoder inputs: {len(nemo_enc_inputs)} items")
        for i, inp in enumerate(nemo_enc_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}")
            else:
                print(f"  input[{i}]: type={type(inp)}")
    
    if nest_enc_inputs:
        print(f"nest encoder inputs: {len(nest_enc_inputs)} items")
        for i, inp in enumerate(nest_enc_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}")
            else:
                print(f"  input[{i}]: type={type(inp)}")
    
    # 3. Check pre_encode input
    print("\n" + "="*60)
    print("3. Pre_encode Forward Inputs")
    print("="*60)
    
    nemo_pre_inputs = nemo_layers.get('encoder.pre_encode', {}).get('forward_inputs')
    nest_pre_inputs = nest_layers.get('encoder.pre_encode', {}).get('forward_inputs')
    
    if nemo_pre_inputs:
        print(f"NeMo pre_encode inputs: {len(nemo_pre_inputs)} items")
        for i, inp in enumerate(nemo_pre_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}")
                print(f"    min={inp.min().item():.6f}, max={inp.max().item():.6f}, mean={inp.mean().item():.6f}")
            else:
                print(f"  input[{i}]: type={type(inp)}, value={inp}")
    
    if nest_pre_inputs:
        print(f"nest pre_encode inputs: {len(nest_pre_inputs)} items")
        for i, inp in enumerate(nest_pre_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}")
                print(f"    min={inp.min().item():.6f}, max={inp.max().item():.6f}, mean={inp.mean().item():.6f}")
            else:
                print(f"  input[{i}]: type={type(inp)}, value={inp}")
    
    # Compare pre_encode inputs
    if nemo_pre_inputs and nest_pre_inputs:
        print("\n  Comparing pre_encode inputs:")
        for i, (ni, si) in enumerate(zip(nemo_pre_inputs, nest_pre_inputs)):
            if isinstance(ni, torch.Tensor) and isinstance(si, torch.Tensor):
                if ni.shape == si.shape:
                    diff = (ni - si).abs()
                    print(f"    input[{i}]: max_diff={diff.max().item():.6e}, mean_diff={diff.mean().item():.6e}")
                else:
                    print(f"    input[{i}]: Shape mismatch! NeMo={ni.shape}, nest={si.shape}")
    
    # 4. Check pre_encode.conv input (after unsqueeze)
    print("\n" + "="*60)
    print("4. Pre_encode.conv Forward Inputs (after unsqueeze)")
    print("="*60)
    
    nemo_conv_inputs = nemo_layers.get('encoder.pre_encode.conv', {}).get('forward_inputs')
    nest_conv_inputs = nest_layers.get('encoder.pre_encode.conv', {}).get('forward_inputs')
    
    if nemo_conv_inputs:
        print(f"NeMo pre_encode.conv inputs: {len(nemo_conv_inputs)} items")
        for i, inp in enumerate(nemo_conv_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}")
                print(f"    min={inp.min().item():.6f}, max={inp.max().item():.6f}, mean={inp.mean().item():.6f}")
    
    if nest_conv_inputs:
        print(f"nest pre_encode.conv inputs: {len(nest_conv_inputs)} items")
        for i, inp in enumerate(nest_conv_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}")
                print(f"    min={inp.min().item():.6f}, max={inp.max().item():.6f}, mean={inp.mean().item():.6f}")
    
    # Compare conv inputs
    if nemo_conv_inputs and nest_conv_inputs:
        print("\n  Comparing pre_encode.conv inputs:")
        for i, (ni, si) in enumerate(zip(nemo_conv_inputs, nest_conv_inputs)):
            if isinstance(ni, torch.Tensor) and isinstance(si, torch.Tensor):
                if ni.shape == si.shape:
                    diff = (ni - si).abs()
                    print(f"    input[{i}]: max_diff={diff.max().item():.6e}, mean_diff={diff.mean().item():.6e}")
                    
                    if diff.max().item() > 1e-5:
                        # Find where they differ
                        max_idx = diff.argmax()
                        idx = []
                        flat_idx = max_idx.item()
                        for dim in reversed(ni.shape):
                            idx.insert(0, flat_idx % dim)
                            flat_idx //= dim
                        print(f"    Max diff at {tuple(idx)}: NeMo={ni[tuple(idx)].item():.6f}, nest={si[tuple(idx)].item():.6f}")
                        
                        # Check if the difference is due to masking (zeros in different places)
                        nemo_zeros = (ni == 0).sum().item()
                        nest_zeros = (si == 0).sum().item()
                        print(f"    NeMo zeros: {nemo_zeros}, nest zeros: {nest_zeros}")
                else:
                    print(f"    input[{i}]: Shape mismatch! NeMo={ni.shape}, nest={si.shape}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

