#!/usr/bin/env python3
"""
Detailed comparison of featurizer outputs.
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
    print("DETAILED FEATURIZER COMPARISON")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get featurizer outputs
    nemo_feat = nemo_layers.get('preprocessor.featurizer', {})
    nest_feat = nest_layers.get('preprocessor.featurizer', {})
    
    nemo_out = nemo_feat.get('forward_outputs', [])
    nest_out = nest_feat.get('forward_outputs', [])
    
    nemo_spec = nemo_out[0]
    nest_spec = nest_out[0]
    nemo_len = nemo_out[1]
    nest_len = nest_out[1]
    
    print(f"\n1. Basic info:")
    print(f"   NeMo shape: {nemo_spec.shape}, seq_len: {nemo_len.tolist()}")
    print(f"   nest shape: {nest_spec.shape}, seq_len: {nest_len.tolist()}")
    
    # Overall diff
    diff = (nemo_spec - nest_spec).abs()
    print(f"\n2. Overall difference:")
    print(f"   Max diff: {diff.max().item():.6e}")
    print(f"   Mean diff: {diff.mean().item():.6e}")
    print(f"   Std diff: {diff.std().item():.6e}")
    
    # Per-sample analysis
    print(f"\n3. Per-sample analysis:")
    batch_size = nemo_spec.shape[0]
    
    for b in range(batch_size):
        valid_len = nemo_len[b].item()
        
        # Valid region
        nemo_valid = nemo_spec[b, :, :valid_len]
        nest_valid = nest_spec[b, :, :valid_len]
        valid_diff = (nemo_valid - nest_valid).abs()
        
        # Padding region
        if valid_len < nemo_spec.shape[-1]:
            nemo_pad = nemo_spec[b, :, valid_len:]
            nest_pad = nest_spec[b, :, valid_len:]
            pad_diff = (nemo_pad - nest_pad).abs()
        else:
            pad_diff = None
        
        print(f"\n   Sample {b} (valid_len={valid_len}, total_len={nemo_spec.shape[-1]}):")
        print(f"     Valid region:")
        print(f"       Max diff: {valid_diff.max().item():.6e}")
        print(f"       Mean diff: {valid_diff.mean().item():.6e}")
        
        # Find max diff location in valid region
        max_idx = valid_diff.argmax()
        feat_idx = (max_idx // valid_len).item()
        time_idx = (max_idx % valid_len).item()
        print(f"       Max diff at: feat={feat_idx}, time={time_idx}")
        print(f"       NeMo value: {nemo_valid[feat_idx, time_idx].item():.8f}")
        print(f"       nest value: {nest_valid[feat_idx, time_idx].item():.8f}")
        
        if pad_diff is not None:
            print(f"     Padding region:")
            print(f"       Max diff: {pad_diff.max().item():.6e}")
            print(f"       NeMo pad values: min={nemo_pad.min().item():.6f}, max={nemo_pad.max().item():.6f}")
            print(f"       nest pad values: min={nest_pad.min().item():.6f}, max={nest_pad.max().item():.6f}")
    
    # Check boundary frames
    print(f"\n4. Boundary frame analysis (Sample 0):")
    valid_len = nemo_len[0].item()
    
    # First frame
    first_diff = (nemo_spec[0, :, 0] - nest_spec[0, :, 0]).abs()
    print(f"   First frame (t=0):")
    print(f"     Max diff: {first_diff.max().item():.6e}")
    print(f"     Mean diff: {first_diff.mean().item():.6e}")
    
    # Last valid frame
    last_valid_diff = (nemo_spec[0, :, valid_len-1] - nest_spec[0, :, valid_len-1]).abs()
    print(f"   Last valid frame (t={valid_len-1}):")
    print(f"     Max diff: {last_valid_diff.max().item():.6e}")
    print(f"     Mean diff: {last_valid_diff.mean().item():.6e}")
    
    # Middle frame
    mid_t = valid_len // 2
    mid_diff = (nemo_spec[0, :, mid_t] - nest_spec[0, :, mid_t]).abs()
    print(f"   Middle frame (t={mid_t}):")
    print(f"     Max diff: {mid_diff.max().item():.6e}")
    print(f"     Mean diff: {mid_diff.mean().item():.6e}")
    
    # Check if difference is in specific frequency bands
    print(f"\n5. Frequency band analysis (Sample 0, valid region):")
    nemo_valid = nemo_spec[0, :, :valid_len]
    nest_valid = nest_spec[0, :, :valid_len]
    
    # Low freq (0-26)
    low_diff = (nemo_valid[:27, :] - nest_valid[:27, :]).abs()
    print(f"   Low freq (0-26): max={low_diff.max().item():.6e}, mean={low_diff.mean().item():.6e}")
    
    # Mid freq (27-53)
    mid_diff = (nemo_valid[27:54, :] - nest_valid[27:54, :]).abs()
    print(f"   Mid freq (27-53): max={mid_diff.max().item():.6e}, mean={mid_diff.mean().item():.6e}")
    
    # High freq (54-79)
    high_diff = (nemo_valid[54:, :] - nest_valid[54:, :]).abs()
    print(f"   High freq (54-79): max={high_diff.max().item():.6e}, mean={high_diff.mean().item():.6e}")
    
    # Check time distribution of differences
    print(f"\n6. Time distribution of differences (Sample 0):")
    time_max_diff = (nemo_valid - nest_valid).abs().max(dim=0)[0]  # Max diff per time step
    
    # Find time steps with largest differences
    top_k = 10
    top_indices = time_max_diff.argsort(descending=True)[:top_k]
    print(f"   Top {top_k} time steps with largest max diff:")
    for i, t in enumerate(top_indices):
        print(f"     t={t.item()}: max_diff={time_max_diff[t].item():.6e}")
    
    # Check if differences are at boundaries
    boundary_times = [0, 1, 2, valid_len-3, valid_len-2, valid_len-1]
    print(f"\n   Boundary time steps:")
    for t in boundary_times:
        if t < valid_len:
            print(f"     t={t}: max_diff={time_max_diff[t].item():.6e}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

