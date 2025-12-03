#!/usr/bin/env python3
"""
Diagnose the first few encoder layers in detail.
"""

import torch
import pickle
from pathlib import Path
import argparse


def compare_tensor(name, t1, t2):
    """Compare two tensors."""
    if t1 is None or t2 is None:
        print(f"  {name}: One is None")
        return
    
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        print(f"  {name}: Not tensors (types: {type(t1)}, {type(t2)})")
        return
    
    if t1.shape != t2.shape:
        print(f"  [SHAPE] {name}: NeMo={t1.shape}, nest={t2.shape}")
        return
    
    diff = (t1.float() - t2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    status = "[OK]" if max_diff < 1e-5 else "[FAIL]"
    print(f"  {status} {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    
    if max_diff > 1e-5:
        max_idx = diff.argmax()
        idx = []
        flat_idx = max_idx.item()
        for dim in reversed(t1.shape):
            idx.insert(0, flat_idx % dim)
            flat_idx //= dim
        print(f"         Max at {tuple(idx)}: NeMo={t1[tuple(idx)].item():.6f}, nest={t2[tuple(idx)].item():.6f}")


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
    print("ENCODER FIRST LAYERS DIAGNOSIS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # List all encoder-related keys
    nemo_encoder_keys = sorted([k for k in nemo_layers.keys() if 'encoder' in k.lower()])
    nest_encoder_keys = sorted([k for k in nest_layers.keys() if 'encoder' in k.lower()])
    
    print(f"\nNeMo encoder keys ({len(nemo_encoder_keys)}): {nemo_encoder_keys[:20]}...")
    print(f"\nnest encoder keys ({len(nest_encoder_keys)}): {nest_encoder_keys[:20]}...")
    
    # Check specific layers
    layers_to_check = [
        'encoder',
        'encoder.pre_encode',
        'encoder.pre_encode.conv',
        'encoder.pre_encode.conv.0',
        'encoder.pre_encode.conv.1',
        'encoder.pre_encode.out',
        'encoder.pos_enc',
        'encoder.pos_enc.dropout',
        'encoder.layers',
        'encoder.layers.0',
        'encoder.layers.0.feed_forward1',
        'encoder.layers.0.self_attn',
        'encoder.layers.0.conv',
        'encoder.layers.0.feed_forward2',
    ]
    
    for layer_name in layers_to_check:
        print(f"\n{'='*60}")
        print(f"Layer: {layer_name}")
        print(f"{'='*60}")
        
        nemo_exists = layer_name in nemo_layers
        nest_exists = layer_name in nest_layers
        
        print(f"  In NeMo: {nemo_exists}, In nest: {nest_exists}")
        
        if not nemo_exists or not nest_exists:
            continue
        
        nemo_data = nemo_layers[layer_name]
        nest_data = nest_layers[layer_name]
        
        # Check forward inputs
        nemo_inputs = nemo_data.get('forward_inputs')
        nest_inputs = nest_data.get('forward_inputs')
        
        if nemo_inputs and nest_inputs:
            print(f"\n  Forward Inputs:")
            for i, (ni, si) in enumerate(zip(nemo_inputs, nest_inputs)):
                if isinstance(ni, torch.Tensor) and isinstance(si, torch.Tensor):
                    compare_tensor(f"input[{i}]", ni, si)
                elif ni is not None or si is not None:
                    print(f"    input[{i}]: NeMo type={type(ni)}, nest type={type(si)}")
        
        # Check forward outputs
        nemo_outputs = nemo_data.get('forward_outputs')
        nest_outputs = nest_data.get('forward_outputs')
        
        if nemo_outputs is not None and nest_outputs is not None:
            print(f"\n  Forward Outputs:")
            
            if isinstance(nemo_outputs, (list, tuple)) and isinstance(nest_outputs, (list, tuple)):
                for i, (no, so) in enumerate(zip(nemo_outputs, nest_outputs)):
                    if isinstance(no, torch.Tensor) and isinstance(so, torch.Tensor):
                        compare_tensor(f"output[{i}]", no, so)
                    elif no is not None or so is not None:
                        print(f"    output[{i}]: NeMo type={type(no)}, nest type={type(so)}")
            elif isinstance(nemo_outputs, torch.Tensor) and isinstance(nest_outputs, torch.Tensor):
                compare_tensor("output", nemo_outputs, nest_outputs)
            else:
                print(f"    NeMo type={type(nemo_outputs)}, nest type={type(nest_outputs)}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

