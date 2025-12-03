#!/usr/bin/env python3
"""
Final diagnosis: Compare outputs accounting for padding.
"""

import torch
import pickle
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Final mismatch diagnosis")
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
            print(f"Auto-detected step: {args.step}")
        else:
            print("No common steps found.")
            return
    
    print("="*80)
    print("FINAL MISMATCH DIAGNOSIS")
    print("="*80)
    
    nemo_batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
    nest_batch = torch.load(nest_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
    
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    nemo_feat = nemo_layers.get('preprocessor.featurizer', {})
    nest_feat = nest_layers.get('preprocessor.featurizer', {})
    
    nemo_out = nemo_feat.get('forward_output', [None])[0]
    nest_out = nest_feat.get('forward_output', [None])[0]
    nemo_len = nemo_feat.get('forward_output', [None, None])[1]
    nest_len = nest_feat.get('forward_output', [None, None])[1]
    
    print(f"\n1. Featurizer output shapes:")
    print(f"   NeMo: {nemo_out.shape if isinstance(nemo_out, torch.Tensor) else 'None'}")
    print(f"   nest: {nest_out.shape if isinstance(nest_out, torch.Tensor) else 'None'}")
    
    print(f"\n2. Sequence lengths:")
    print(f"   NeMo: {nemo_len}")
    print(f"   nest: {nest_len}")
    
    if isinstance(nemo_out, torch.Tensor) and isinstance(nest_out, torch.Tensor):
        full_diff = (nemo_out - nest_out).abs()
        print(f"\n3. Full output comparison:")
        print(f"   Max diff: {full_diff.max().item():.6e}")
        print(f"   Mean diff: {full_diff.mean().item():.6e}")
        
        print(f"\n4. Per-sample comparison (within valid length):")
        batch_size = nemo_out.shape[0]
        
        for b in range(batch_size):
            valid_len = min(nemo_len[b].item(), nest_len[b].item())
            nemo_valid = nemo_out[b, :, :valid_len]
            nest_valid = nest_out[b, :, :valid_len]
            
            diff = (nemo_valid - nest_valid).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"\n   Sample {b} (valid_len={valid_len}):")
            print(f"     Max diff: {max_diff:.6e}")
            print(f"     Mean diff: {mean_diff:.6e}")
            
            if valid_len > 10:
                first_10_diff = (nemo_valid[:, :10] - nest_valid[:, :10]).abs().max().item()
                last_10_diff = (nemo_valid[:, -10:] - nest_valid[:, -10:]).abs().max().item()
                middle_diff = (nemo_valid[:, 10:-10] - nest_valid[:, 10:-10]).abs().max().item()
                
                print(f"     First 10 frames: {first_10_diff:.6e}")
                print(f"     Middle frames: {middle_diff:.6e}")
                print(f"     Last 10 frames: {last_10_diff:.6e}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

