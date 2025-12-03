#!/usr/bin/env python3
"""
Diagnose encoder layer differences step by step.
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
    print("ENCODER LAYER DIAGNOSIS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get sorted keys for encoder layers
    nemo_keys = sorted([k for k in nemo_layers.keys() if 'encoder' in k])
    
    print(f"Found {len(nemo_keys)} encoder layers.")
    
    # Filter for main blocks to avoid too much noise
    main_layers = [
        'encoder.pos_enc', 
        'encoder.pre_encode',
        'encoder.layers.0',
        'encoder.layers.0.self_attn',
        'encoder.layers.0.feed_forward1',
        'encoder.layers.0.conv',
        'encoder.layers.0.feed_forward2',
        'encoder.layers.0.norm',
    ]
    
    # Check all layers in order
    for layer_name in nemo_keys:
        if layer_name not in nest_layers:
            print(f"[MISSING] {layer_name} not found in nest")
            continue
            
        nemo_out = nemo_layers[layer_name].get('forward_outputs')
        nest_out = nest_layers[layer_name].get('forward_outputs')
        
        if nemo_out is None or nest_out is None:
            continue
            
        # Handle list/tuple outputs (e.g. (output, length))
        if isinstance(nemo_out, (list, tuple)):
            nemo_tensor = nemo_out[0]
        else:
            nemo_tensor = nemo_out
            
        if isinstance(nest_out, (list, tuple)):
            nest_tensor = nest_out[0]
        else:
            nest_tensor = nest_out
            
        if not isinstance(nemo_tensor, torch.Tensor) or not isinstance(nest_tensor, torch.Tensor):
            continue
            
        if nemo_tensor.shape != nest_tensor.shape:
            print(f"[SHAPE] {layer_name}: NeMo={nemo_tensor.shape}, nest={nest_tensor.shape}")
            continue
            
        diff = (nemo_tensor - nest_tensor).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        status = "[OK]" if max_diff < 1e-4 else "[FAIL]"
        print(f"{status} {layer_name:<40} max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        
        # If failed, show more details
        if max_diff > 1e-4:
            # Find max diff index
            max_idx = diff.argmax()
            idx = []
            flat_idx = max_idx.item()
            for dim in reversed(nemo_tensor.shape):
                idx.insert(0, flat_idx % dim)
                flat_idx //= dim
            
            print(f"       Max diff at {tuple(idx)}: NeMo={nemo_tensor[tuple(idx)].item():.6f}, nest={nest_tensor[tuple(idx)].item():.6f}")
            
            # If it's the first failure, stop (optional)
            # break 

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

