#!/usr/bin/env python3
"""
Diagnose boundary frame differences between NeMo and nest.
Focus on time=0 and time=valid_len-1 frames.
"""

import torch
import pickle
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Diagnose boundary frames")
    parser.add_argument("--nemo_dir", type=str, required=True, help="NeMo output directory")
    parser.add_argument("--nest_dir", type=str, required=True, help="nest output directory")
    parser.add_argument("--step", type=int, default=None, help="Step to compare")
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir)
    nest_dir = Path(args.nest_dir)
    
    # Auto-detect step
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
    print("DIAGNOSE BOUNDARY FRAMES")
    print("="*80)
    
    # Load layer outputs
    try:
        with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
            nemo_layers = pickle.load(f)
        with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
            nest_layers = pickle.load(f)
    except Exception as e:
        print(f"Error loading layer outputs: {e}")
        return
    
    # Get featurizer outputs
    nemo_feat = nemo_layers.get('preprocessor.featurizer') or nemo_layers.get('preprocessor')
    nest_feat = nest_layers.get('preprocessor.featurizer') or nest_layers.get('preprocessor')
    
    if not nemo_feat or not nest_feat:
        print("Featurizer not found")
        return
    
    nemo_out = nemo_feat['forward_outputs'][0]  # [B, D, T]
    nest_out = nest_feat['forward_outputs'][0]
    nemo_seq_len = nemo_feat['forward_outputs'][1]
    nest_seq_len = nest_feat['forward_outputs'][1]
    
    batch_size = nemo_out.shape[0]
    n_features = nemo_out.shape[1]
    
    for b in range(batch_size):
        print(f"\n{'='*60}")
        print(f"Sample {b}:")
        print(f"{'='*60}")
        
        valid_len = nemo_seq_len[b].item()
        
        nemo_sample = nemo_out[b]  # [D, T]
        nest_sample = nest_out[b]
        
        # Compare first frame (time=0)
        print(f"\n1. First Frame (time=0):")
        nemo_first = nemo_sample[:, 0]
        nest_first = nest_sample[:, 0]
        
        diff_first = (nemo_first - nest_first).abs()
        print(f"   Max diff: {diff_first.max().item():.6e}")
        print(f"   Mean diff: {diff_first.mean().item():.6e}")
        
        # Compare last valid frame (time=valid_len-1)
        print(f"\n2. Last Valid Frame (time={valid_len-1}):")
        nemo_last = nemo_sample[:, valid_len-1]
        nest_last = nest_sample[:, valid_len-1]
        
        diff_last = (nemo_last - nest_last).abs()
        print(f"   Max diff: {diff_last.max().item():.6e}")
        print(f"   Mean diff: {diff_last.mean().item():.6e}")
        
        # Compare middle frames
        mid_start = valid_len // 4
        mid_end = 3 * valid_len // 4
        print(f"\n3. Middle Frames (time={mid_start} to {mid_end}):")
        nemo_mid = nemo_sample[:, mid_start:mid_end]
        nest_mid = nest_sample[:, mid_start:mid_end]
        
        diff_mid = (nemo_mid - nest_mid).abs()
        print(f"   Max diff: {diff_mid.max().item():.6e}")
        print(f"   Mean diff: {diff_mid.mean().item():.6e}")
        
        # Compare excluding first and last frames
        print(f"\n4. Excluding First and Last Frames (time=1 to {valid_len-2}):")
        if valid_len > 2:
            nemo_inner = nemo_sample[:, 1:valid_len-1]
            nest_inner = nest_sample[:, 1:valid_len-1]
            
            diff_inner = (nemo_inner - nest_inner).abs()
            print(f"   Max diff: {diff_inner.max().item():.6e}")
            print(f"   Mean diff: {diff_inner.mean().item():.6e}")
            
            # Find where max diff occurs in inner region
            max_idx = diff_inner.argmax()
            max_idx_2d = torch.unravel_index(max_idx, diff_inner.shape)
            actual_time = max_idx_2d[1].item() + 1  # +1 because we excluded first frame
            
            print(f"   Max diff at: feature={max_idx_2d[0].item()}, time={actual_time}")
        
        # Check if first frame values exist elsewhere
        print(f"\n5. Value Location Analysis for First Frame:")
        for feat_idx in [34, 75, 35]:  # Top 3 features with highest diff at time=0
            if feat_idx >= n_features:
                continue
            nemo_val = nemo_first[feat_idx].item()
            nest_val = nest_first[feat_idx].item()
            
            # Find where nest's value appears in nemo
            nemo_feature = nemo_sample[feat_idx, :valid_len]
            nest_feature = nest_sample[feat_idx, :valid_len]
            
            nemo_closest_to_nest = (nemo_feature - nest_val).abs().argmin().item()
            nest_closest_to_nemo = (nest_feature - nemo_val).abs().argmin().item()
            
            nemo_closest_val = nemo_feature[nemo_closest_to_nest].item()
            nest_closest_val = nest_feature[nest_closest_to_nemo].item()
            
            print(f"   Feature {feat_idx}:")
            print(f"     NeMo[0]={nemo_val:.6f}, nest[0]={nest_val:.6f}")
            print(f"     NeMo closest to nest's value: time={nemo_closest_to_nest}, val={nemo_closest_val:.6f}, diff={abs(nemo_closest_val-nest_val):.6e}")
            print(f"     nest closest to NeMo's value: time={nest_closest_to_nemo}, val={nest_closest_val:.6f}, diff={abs(nest_closest_val-nemo_val):.6e}")
        
        # Check normalization statistics
        print(f"\n6. Normalization Statistics (valid region):")
        nemo_valid = nemo_sample[:, :valid_len]
        nest_valid = nest_sample[:, :valid_len]
        
        # Per-feature mean and std
        nemo_mean = nemo_valid.mean(dim=1)
        nest_mean = nest_valid.mean(dim=1)
        nemo_std = nemo_valid.std(dim=1)
        nest_std = nest_valid.std(dim=1)
        
        mean_diff = (nemo_mean - nest_mean).abs()
        std_diff = (nemo_std - nest_std).abs()
        
        print(f"   Mean diff (per-feature): max={mean_diff.max().item():.6e}, avg={mean_diff.mean().item():.6e}")
        print(f"   Std diff (per-feature): max={std_diff.max().item():.6e}, avg={std_diff.mean().item():.6e}")
        
        # Check if the difference is due to different mean/std calculation
        print(f"\n7. Re-normalize nest with NeMo's mean/std:")
        # Reverse nest's normalization and apply NeMo's
        # This is approximate since we don't have the original mean/std
        
        # Check correlation at first frame
        corr_first = torch.corrcoef(torch.stack([nemo_first, nest_first]))[0, 1].item()
        print(f"   Correlation at first frame: {corr_first:.6f}")
    
    print("\n" + "="*80)
    print("Diagnosis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

