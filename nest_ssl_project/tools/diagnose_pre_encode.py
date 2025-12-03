#!/usr/bin/env python3
"""
Diagnose pre_encode (ConvSubsampling) layer differences.
"""

import torch
import pickle
from pathlib import Path
import argparse


def compare_tensors(name, t1, t2, prefix=""):
    """Compare two tensors and print detailed info."""
    if t1 is None or t2 is None:
        print(f"{prefix}{name}: One is None")
        return
    
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        print(f"{prefix}{name}: Not tensors")
        return
    
    if t1.shape != t2.shape:
        print(f"{prefix}{name}: Shape mismatch: {t1.shape} vs {t2.shape}")
        return
    
    diff = (t1.float() - t2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.float().mean().item()
    
    status = "[OK]" if max_diff < 1e-5 else "[FAIL]"
    print(f"{prefix}{status} {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, shape={t1.shape}")
    
    if max_diff > 1e-5:
        # Find max diff location
        max_idx = diff.argmax()
        idx = []
        flat_idx = max_idx.item()
        for dim in reversed(t1.shape):
            idx.insert(0, flat_idx % dim)
            flat_idx //= dim
        print(f"{prefix}       Max diff at {tuple(idx)}: NeMo={t1[tuple(idx)].item():.6f}, nest={t2[tuple(idx)].item():.6f}")
        
        # Check for zeros
        nemo_zeros = (t1 == 0).sum().item()
        nest_zeros = (t2 == 0).sum().item()
        total = t1.numel()
        print(f"{prefix}       NeMo zeros: {nemo_zeros}/{total} ({100*nemo_zeros/total:.2f}%)")
        print(f"{prefix}       nest zeros: {nest_zeros}/{total} ({100*nest_zeros/total:.2f}%)")
        
        # Check if zeros are at different positions
        nemo_zero_mask = (t1 == 0)
        nest_zero_mask = (t2 == 0)
        diff_zeros = (nemo_zero_mask != nest_zero_mask).sum().item()
        print(f"{prefix}       Positions with different zero status: {diff_zeros}")


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
    print("PRE_ENCODE (ConvSubsampling) DIAGNOSIS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Check pre_encode input (should be preprocessor output)
    print("\n" + "="*40)
    print("1. CHECK INPUT TO PRE_ENCODE")
    print("="*40)
    
    # The input to pre_encode is the output of preprocessor
    if 'preprocessor' in nemo_layers and 'preprocessor' in nest_layers:
        nemo_prep = nemo_layers['preprocessor'].get('forward_outputs')
        nest_prep = nest_layers['preprocessor'].get('forward_outputs')
        
        if isinstance(nemo_prep, (list, tuple)):
            nemo_prep_out = nemo_prep[0]
        else:
            nemo_prep_out = nemo_prep
            
        if isinstance(nest_prep, (list, tuple)):
            nest_prep_out = nest_prep[0]
        else:
            nest_prep_out = nest_prep
            
        compare_tensors("preprocessor output (input to encoder)", nemo_prep_out, nest_prep_out)
    
    # Check pre_encode.conv input
    print("\n" + "="*40)
    print("2. CHECK PRE_ENCODE.CONV LAYERS")
    print("="*40)
    
    pre_encode_keys = sorted([k for k in nemo_layers.keys() if 'pre_encode' in k])
    print(f"Found {len(pre_encode_keys)} pre_encode layers: {pre_encode_keys}")
    
    for key in pre_encode_keys:
        if key not in nest_layers:
            print(f"[MISSING] {key} not in nest")
            continue
            
        print(f"\n--- {key} ---")
        
        # Check forward inputs
        nemo_inputs = nemo_layers[key].get('forward_inputs')
        nest_inputs = nest_layers[key].get('forward_inputs')
        
        if nemo_inputs and nest_inputs:
            for i, (ni, si) in enumerate(zip(nemo_inputs, nest_inputs)):
                if isinstance(ni, torch.Tensor) and isinstance(si, torch.Tensor):
                    compare_tensors(f"input[{i}]", ni, si, prefix="  ")
        
        # Check forward outputs
        nemo_outputs = nemo_layers[key].get('forward_outputs')
        nest_outputs = nest_layers[key].get('forward_outputs')
        
        if isinstance(nemo_outputs, (list, tuple)):
            for i, (no, so) in enumerate(zip(nemo_outputs, nest_outputs)):
                if isinstance(no, torch.Tensor) and isinstance(so, torch.Tensor):
                    compare_tensors(f"output[{i}]", no, so, prefix="  ")
        elif isinstance(nemo_outputs, torch.Tensor) and isinstance(nest_outputs, torch.Tensor):
            compare_tensors("output", nemo_outputs, nest_outputs, prefix="  ")
    
    # Check pos_enc
    print("\n" + "="*40)
    print("3. CHECK POS_ENC")
    print("="*40)
    
    pos_enc_keys = sorted([k for k in nemo_layers.keys() if 'pos_enc' in k and 'linear_pos' not in k])
    for key in pos_enc_keys:
        if key not in nest_layers:
            print(f"[MISSING] {key} not in nest")
            continue
            
        print(f"\n--- {key} ---")
        
        nemo_outputs = nemo_layers[key].get('forward_outputs')
        nest_outputs = nest_layers[key].get('forward_outputs')
        
        if isinstance(nemo_outputs, (list, tuple)):
            for i, (no, so) in enumerate(zip(nemo_outputs, nest_outputs)):
                if isinstance(no, torch.Tensor) and isinstance(so, torch.Tensor):
                    compare_tensors(f"output[{i}]", no, so, prefix="  ")
        elif isinstance(nemo_outputs, torch.Tensor) and isinstance(nest_outputs, torch.Tensor):
            compare_tensors("output", nemo_outputs, nest_outputs, prefix="  ")
    
    # Check weights
    print("\n" + "="*40)
    print("4. CHECK PRE_ENCODE WEIGHTS")
    print("="*40)
    
    nemo_weights_path = nemo_dir / "initial_weights" / "parameter_weights.pt"
    nest_weights_path = nest_dir / "initial_weights" / "parameter_weights.pt"
    
    if nemo_weights_path.exists() and nest_weights_path.exists():
        nemo_weights = torch.load(nemo_weights_path, map_location='cpu', weights_only=True)
        nest_weights = torch.load(nest_weights_path, map_location='cpu', weights_only=True)
        
        pre_encode_weight_keys = sorted([k for k in nemo_weights.keys() if 'pre_encode' in k])
        print(f"Found {len(pre_encode_weight_keys)} pre_encode weight keys")
        
        for key in pre_encode_weight_keys[:10]:  # First 10 only
            if key in nest_weights:
                compare_tensors(key, nemo_weights[key], nest_weights[key], prefix="  ")
            else:
                print(f"  [MISSING] {key} not in nest weights")
    else:
        print("Weight files not found")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

