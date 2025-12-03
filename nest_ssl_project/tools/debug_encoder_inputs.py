#!/usr/bin/env python3
"""
Debug why encoder forward_inputs is None
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
    print("DEBUG ENCODER INPUTS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Check encoder data
    print("\n" + "="*60)
    print("NeMo encoder data")
    print("="*60)
    nemo_enc = nemo_layers.get('encoder', {})
    print(f"Keys: {nemo_enc.keys()}")
    nemo_enc_inputs = nemo_enc.get('forward_inputs')
    print(f"forward_inputs: type={type(nemo_enc_inputs)}, len={len(nemo_enc_inputs) if nemo_enc_inputs else 0}")
    nemo_enc_outputs = nemo_enc.get('forward_outputs')
    print(f"forward_outputs: type={type(nemo_enc_outputs)}, len={len(nemo_enc_outputs) if isinstance(nemo_enc_outputs, (list, tuple)) else 'N/A'}")
    
    if nemo_enc_inputs and len(nemo_enc_inputs) > 0:
        print("  Inputs:")
        for i, inp in enumerate(nemo_enc_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"    input[{i}]: shape={inp.shape}, mean={inp.float().mean().item():.6f}")
            elif inp is None:
                print(f"    input[{i}]: None")
            else:
                print(f"    input[{i}]: type={type(inp)}")
    else:
        print("  Inputs: EMPTY or None")
    
    print("\n" + "="*60)
    print("nest encoder data")
    print("="*60)
    nest_enc = nest_layers.get('encoder', {})
    print(f"Keys: {nest_enc.keys()}")
    nest_enc_inputs = nest_enc.get('forward_inputs')
    print(f"forward_inputs: type={type(nest_enc_inputs)}, len={len(nest_enc_inputs) if nest_enc_inputs else 0}")
    nest_enc_outputs = nest_enc.get('forward_outputs')
    print(f"forward_outputs: type={type(nest_enc_outputs)}, len={len(nest_enc_outputs) if isinstance(nest_enc_outputs, (list, tuple)) else 'N/A'}")
    
    if nest_enc_inputs and len(nest_enc_inputs) > 0:
        print("  Inputs:")
        for i, inp in enumerate(nest_enc_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"    input[{i}]: shape={inp.shape}, mean={inp.float().mean().item():.6f}")
            elif inp is None:
                print(f"    input[{i}]: None")
            else:
                print(f"    input[{i}]: type={type(inp)}")
    else:
        print("  Inputs: EMPTY or None")
    
    # Check pre_encode data
    print("\n" + "="*60)
    print("NeMo pre_encode data")
    print("="*60)
    nemo_pre = nemo_layers.get('encoder.pre_encode', {})
    print(f"Keys: {nemo_pre.keys()}")
    nemo_pre_inputs = nemo_pre.get('forward_inputs')
    print(f"forward_inputs: type={type(nemo_pre_inputs)}, len={len(nemo_pre_inputs) if nemo_pre_inputs else 0}")
    
    if nemo_pre_inputs and len(nemo_pre_inputs) > 0:
        print("  Inputs:")
        for i, inp in enumerate(nemo_pre_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"    input[{i}]: shape={inp.shape}, mean={inp.float().mean().item():.6f}")
            elif inp is None:
                print(f"    input[{i}]: None")
            else:
                print(f"    input[{i}]: type={type(inp)}")
    else:
        print("  Inputs: EMPTY or None")
    
    print("\n" + "="*60)
    print("nest pre_encode data")
    print("="*60)
    nest_pre = nest_layers.get('encoder.pre_encode', {})
    print(f"Keys: {nest_pre.keys()}")
    nest_pre_inputs = nest_pre.get('forward_inputs')
    print(f"forward_inputs: type={type(nest_pre_inputs)}, len={len(nest_pre_inputs) if nest_pre_inputs else 0}")
    
    if nest_pre_inputs and len(nest_pre_inputs) > 0:
        print("  Inputs:")
        for i, inp in enumerate(nest_pre_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"    input[{i}]: shape={inp.shape}, mean={inp.float().mean().item():.6f}")
            elif inp is None:
                print(f"    input[{i}]: None")
            else:
                print(f"    input[{i}]: type={type(inp)}")
    else:
        print("  Inputs: EMPTY or None")
    
    # Compare pre_encode inputs if both exist
    if nemo_pre.get('forward_inputs') and nest_pre.get('forward_inputs'):
        print("\n" + "="*60)
        print("Comparing pre_encode inputs")
        print("="*60)
        
        for i, (ni, si) in enumerate(zip(nemo_pre['forward_inputs'], nest_pre['forward_inputs'])):
            if isinstance(ni, torch.Tensor) and isinstance(si, torch.Tensor):
                if ni.shape == si.shape:
                    diff = (ni.float() - si.float()).abs()
                    status = "[OK]" if diff.max().item() < 1e-5 else "[FAIL]"
                    print(f"{status} input[{i}]: max_diff={diff.max().item():.6e}")
                else:
                    print(f"[SHAPE] input[{i}]: NeMo={ni.shape}, nest={si.shape}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

