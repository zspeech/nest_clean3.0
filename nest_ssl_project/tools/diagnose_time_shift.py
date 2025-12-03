#!/usr/bin/env python3
"""
Diagnose time shift between NeMo and nest featurizer outputs.
"""

import torch
import pickle
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Diagnose time shift")
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
    print("DIAGNOSE TIME SHIFT")
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
    
    print(f"\n1. Output shapes:")
    print(f"   NeMo: {nemo_out.shape}, seq_len: {nemo_seq_len.tolist()}")
    print(f"   nest: {nest_out.shape}, seq_len: {nest_seq_len.tolist()}")
    
    # Check sample by sample
    batch_size = nemo_out.shape[0]
    
    for b in range(batch_size):
        print(f"\n{'='*60}")
        print(f"Sample {b}:")
        print(f"{'='*60}")
        
        valid_len_nemo = nemo_seq_len[b].item()
        valid_len_nest = nest_seq_len[b].item()
        
        print(f"   valid_len: NeMo={valid_len_nemo}, nest={valid_len_nest}")
        
        nemo_sample = nemo_out[b]  # [D, T]
        nest_sample = nest_out[b]
        
        # Compare valid regions
        if valid_len_nemo == valid_len_nest:
            valid_len = valid_len_nemo
            nemo_valid = nemo_sample[:, :valid_len]
            nest_valid = nest_sample[:, :valid_len]
            
            diff = (nemo_valid - nest_valid).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            max_idx = diff.argmax()
            max_idx_2d = torch.unravel_index(max_idx, nemo_valid.shape)
            
            print(f"\n   Valid region comparison:")
            print(f"     Max diff: {max_diff:.6e}")
            print(f"     Mean diff: {mean_diff:.6e}")
            print(f"     Max diff at: feature={max_idx_2d[0].item()}, time={max_idx_2d[1].item()}")
            
            # Check if it's a time shift
            print(f"\n   Checking for time shift...")
            
            # Try different shifts
            best_shift = 0
            best_diff = max_diff
            
            for shift in range(-5, 6):
                if shift == 0:
                    continue
                
                if shift > 0:
                    # nest is shifted right relative to nemo
                    nemo_shifted = nemo_valid[:, shift:]
                    nest_shifted = nest_valid[:, :-shift]
                else:
                    # nest is shifted left relative to nemo
                    nemo_shifted = nemo_valid[:, :shift]
                    nest_shifted = nest_valid[:, -shift:]
                
                if nemo_shifted.shape[1] == 0:
                    continue
                
                shifted_diff = (nemo_shifted - nest_shifted).abs().max().item()
                if shifted_diff < best_diff:
                    best_diff = shifted_diff
                    best_shift = shift
            
            if best_shift != 0:
                print(f"     Best shift: {best_shift} frames")
                print(f"     Diff after shift: {best_diff:.6e}")
            else:
                print(f"     No time shift detected")
            
            # Check feature-by-feature
            print(f"\n   Feature-by-feature analysis (first 10 features with highest diff):")
            feature_diffs = diff.max(dim=1)[0]  # Max diff per feature
            top_features = feature_diffs.topk(min(10, feature_diffs.shape[0]))
            
            for i, (val, idx) in enumerate(zip(top_features.values, top_features.indices)):
                feat_idx = idx.item()
                feat_diff = val.item()
                
                # Find time of max diff for this feature
                time_idx = diff[feat_idx].argmax().item()
                nemo_val = nemo_valid[feat_idx, time_idx].item()
                nest_val = nest_valid[feat_idx, time_idx].item()
                
                print(f"     Feature {feat_idx}: max_diff={feat_diff:.6e} at time={time_idx}")
                print(f"       NeMo value: {nemo_val:.6f}, nest value: {nest_val:.6f}")
            
            # Check if values at specific positions are swapped
            print(f"\n   Checking for value swap pattern...")
            
            # Find positions where NeMo has nest's max diff value and vice versa
            nemo_flat = nemo_valid.flatten()
            nest_flat = nest_valid.flatten()
            
            # Get the value at max diff position
            max_pos = max_idx.item()
            nemo_val_at_max = nemo_flat[max_pos].item()
            nest_val_at_max = nest_flat[max_pos].item()
            
            # Search for these values in opposite tensors
            nemo_has_nest_val = (nemo_flat - nest_val_at_max).abs().min().item()
            nest_has_nemo_val = (nest_flat - nemo_val_at_max).abs().min().item()
            
            print(f"     At max diff position: NeMo={nemo_val_at_max:.6f}, nest={nest_val_at_max:.6f}")
            print(f"     NeMo has nest's value within: {nemo_has_nest_val:.6e}")
            print(f"     nest has NeMo's value within: {nest_has_nemo_val:.6e}")
            
            # Check correlation
            correlation = torch.corrcoef(torch.stack([nemo_flat, nest_flat]))[0, 1].item()
            print(f"\n   Correlation: {correlation:.6f}")
    
    print("\n" + "="*80)
    print("Diagnosis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

