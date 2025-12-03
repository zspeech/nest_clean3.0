#!/usr/bin/env python3
"""
Check if noisy_audio in batch is the same between NeMo and nest
"""

import torch
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
    print("BATCH NOISY AUDIO CHECK")
    print("="*80)
    
    # Load batch data
    nemo_batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", map_location='cpu', weights_only=False)
    nest_batch = torch.load(nest_dir / f"step_{args.step}" / "batch.pt", map_location='cpu', weights_only=False)
    
    print(f"\nNeMo batch keys: {nemo_batch.keys()}")
    print(f"nest batch keys: {nest_batch.keys()}")
    
    # Check each field
    fields_to_check = ['audio', 'audio_len', 'noise', 'noise_len', 'noisy_audio', 'noisy_audio_len']
    
    for field in fields_to_check:
        print(f"\n{'='*60}")
        print(f"Field: {field}")
        print(f"{'='*60}")
        
        nemo_val = nemo_batch.get(field)
        nest_val = nest_batch.get(field)
        
        if nemo_val is None or nest_val is None:
            print(f"  Missing: NeMo={nemo_val is None}, nest={nest_val is None}")
            continue
        
        if isinstance(nemo_val, torch.Tensor) and isinstance(nest_val, torch.Tensor):
            print(f"  NeMo: shape={nemo_val.shape}, dtype={nemo_val.dtype}")
            print(f"  nest: shape={nest_val.shape}, dtype={nest_val.dtype}")
            
            if nemo_val.shape == nest_val.shape:
                diff = (nemo_val.float() - nest_val.float()).abs()
                print(f"  Max diff: {diff.max().item():.6e}")
                print(f"  Mean diff: {diff.mean().item():.6e}")
                
                if diff.max().item() > 1e-5:
                    print(f"  [FAIL] Values differ!")
                    # Show some sample values
                    print(f"  NeMo first 5: {nemo_val.flatten()[:5].tolist()}")
                    print(f"  nest first 5: {nest_val.flatten()[:5].tolist()}")
                else:
                    print(f"  [OK] Values match!")
            else:
                print(f"  [FAIL] Shape mismatch!")
        else:
            print(f"  NeMo type: {type(nemo_val)}")
            print(f"  nest type: {type(nest_val)}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

