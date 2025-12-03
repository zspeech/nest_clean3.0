#!/usr/bin/env python3
"""
Check all preprocessor calls to understand the data flow
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
    print("ALL PREPROCESSOR CALLS CHECK")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Check preprocessor
    print("\n" + "="*60)
    print("NeMo preprocessor")
    print("="*60)
    
    nemo_prep = nemo_layers.get('preprocessor', {})
    nemo_all_inputs = nemo_prep.get('all_forward_inputs', [])
    nemo_all_outputs = nemo_prep.get('all_forward_outputs', [])
    
    print(f"Total calls: {len(nemo_all_outputs)}")
    
    for i, (inputs, outputs) in enumerate(zip(nemo_all_inputs, nemo_all_outputs)):
        print(f"\n  Call {i}:")
        
        # Inputs
        if inputs:
            print(f"    Inputs: {len(inputs)} items")
            for j, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    mean_val = inp.float().mean().item() if inp.numel() > 0 else 0
                    print(f"      input[{j}]: shape={inp.shape}, mean={mean_val:.6f}")
        
        # Outputs
        if isinstance(outputs, (list, tuple)):
            print(f"    Outputs: {len(outputs)} items")
            for j, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    mean_val = out.float().mean().item() if out.numel() > 0 else 0
                    print(f"      output[{j}]: shape={out.shape}, mean={mean_val:.6f}")
        elif isinstance(outputs, torch.Tensor):
            mean_val = outputs.float().mean().item() if outputs.numel() > 0 else 0
            print(f"    Output: shape={outputs.shape}, mean={mean_val:.6f}")
    
    print("\n" + "="*60)
    print("nest preprocessor")
    print("="*60)
    
    nest_prep = nest_layers.get('preprocessor', {})
    nest_all_inputs = nest_prep.get('all_forward_inputs', [])
    nest_all_outputs = nest_prep.get('all_forward_outputs', [])
    
    print(f"Total calls: {len(nest_all_outputs)}")
    
    for i, (inputs, outputs) in enumerate(zip(nest_all_inputs, nest_all_outputs)):
        print(f"\n  Call {i}:")
        
        # Inputs
        if inputs:
            print(f"    Inputs: {len(inputs)} items")
            for j, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    mean_val = inp.float().mean().item() if inp.numel() > 0 else 0
                    print(f"      input[{j}]: shape={inp.shape}, mean={mean_val:.6f}")
        
        # Outputs
        if isinstance(outputs, (list, tuple)):
            print(f"    Outputs: {len(outputs)} items")
            for j, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    mean_val = out.float().mean().item() if out.numel() > 0 else 0
                    print(f"      output[{j}]: shape={out.shape}, mean={mean_val:.6f}")
        elif isinstance(outputs, torch.Tensor):
            mean_val = outputs.float().mean().item() if outputs.numel() > 0 else 0
            print(f"    Output: shape={outputs.shape}, mean={mean_val:.6f}")
    
    # Compare the second call (which should be used by encoder)
    if len(nemo_all_outputs) >= 2 and len(nest_all_outputs) >= 2:
        print("\n" + "="*60)
        print("Comparing Call 1 (noisy signal, used by encoder)")
        print("="*60)
        
        nemo_out_1 = nemo_all_outputs[1]
        nest_out_1 = nest_all_outputs[1]
        
        if isinstance(nemo_out_1, (list, tuple)):
            nemo_tensor = nemo_out_1[0]
        else:
            nemo_tensor = nemo_out_1
            
        if isinstance(nest_out_1, (list, tuple)):
            nest_tensor = nest_out_1[0]
        else:
            nest_tensor = nest_out_1
        
        if isinstance(nemo_tensor, torch.Tensor) and isinstance(nest_tensor, torch.Tensor):
            if nemo_tensor.shape == nest_tensor.shape:
                diff = (nemo_tensor - nest_tensor).abs()
                status = "[OK]" if diff.max().item() < 1e-5 else "[FAIL]"
                print(f"{status} max_diff={diff.max().item():.6e}, mean_diff={diff.mean().item():.6e}")
                
                if diff.max().item() > 1e-5:
                    # Find max diff location
                    max_idx = diff.argmax()
                    idx = []
                    flat_idx = max_idx.item()
                    for dim in reversed(nemo_tensor.shape):
                        idx.insert(0, flat_idx % dim)
                        flat_idx //= dim
                    print(f"Max diff at {tuple(idx)}: NeMo={nemo_tensor[tuple(idx)].item():.6f}, nest={nest_tensor[tuple(idx)].item():.6f}")
            else:
                print(f"[SHAPE] NeMo={nemo_tensor.shape}, nest={nest_tensor.shape}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

