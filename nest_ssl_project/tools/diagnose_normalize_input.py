#!/usr/bin/env python3
"""
Diagnose the actual input to normalize_batch during NeMo runtime.
This script tries to reverse-engineer the Log Mel input from the normalized output.
"""

import torch
import pickle
import sys
from pathlib import Path
import argparse


def compare_tensors(name, t1, t2, atol=1e-5, rtol=1e-5):
    """Compare two tensors."""
    if t1 is None or t2 is None:
        print(f"  {name}: One is None")
        return False
    
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        print(f"  {name}: Not tensors: {type(t1)} vs {type(t2)}")
        return False
    
    if t1.shape != t2.shape:
        print(f"  {name}: Shape mismatch: {t1.shape} vs {t2.shape}")
        return False
    
    t1_float = t1.float()
    t2_float = t2.float()
    
    diff = (t1_float - t2_float).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    is_close = torch.allclose(t1_float, t2_float, atol=atol, rtol=rtol)
    
    status = "[OK]" if is_close else "[FAIL]"
    print(f"  {name}: {status} Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    
    if not is_close:
        max_idx = diff.argmax()
        max_idx_unraveled = torch.unravel_index(max_idx, t1.shape)
        print(f"    Max diff at index: {max_idx_unraveled}")
        print(f"    t1 value: {t1.flatten()[max_idx].item():.6f}")
        print(f"    t2 value: {t2.flatten()[max_idx].item():.6f}")
    
    return is_close


def main():
    parser = argparse.ArgumentParser(description="Diagnose normalize_batch input")
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
    print("DIAGNOSE NORMALIZE_BATCH INPUT")
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
    
    # Get featurizer outputs (after normalization)
    if 'preprocessor.featurizer' not in nemo_layers or 'preprocessor.featurizer' not in nest_layers:
        # Try just 'preprocessor'
        if 'preprocessor' in nemo_layers and 'preprocessor' in nest_layers:
            nemo_feat = nemo_layers['preprocessor']
            nest_feat = nest_layers['preprocessor']
        else:
            print("Featurizer/preprocessor layer not found")
            print(f"Available NeMo layers: {list(nemo_layers.keys())[:20]}")
            print(f"Available nest layers: {list(nest_layers.keys())[:20]}")
            return
    else:
        nemo_feat = nemo_layers['preprocessor.featurizer']
        nest_feat = nest_layers['preprocessor.featurizer']
    
    nemo_outputs = nemo_feat.get('forward_outputs', [])
    nest_outputs = nest_feat.get('forward_outputs', [])
    
    if not nemo_outputs or not nest_outputs:
        print("No outputs found")
        return
    
    # Get actual normalized output
    nemo_normalized = nemo_outputs[0]  # [B, D, T]
    nest_normalized = nest_outputs[0]
    nemo_seq_len = nemo_outputs[1]
    nest_seq_len = nest_outputs[1]
    
    print(f"\n1. Actual Featurizer Outputs (after normalization):")
    print(f"   NeMo shape: {nemo_normalized.shape}, seq_len: {nemo_seq_len.tolist()}")
    print(f"   nest shape: {nest_normalized.shape}, seq_len: {nest_seq_len.tolist()}")
    
    # Compare actual outputs
    print(f"\n2. Comparing Actual Featurizer Outputs:")
    compare_tensors("NeMo vs nest (actual output)", nemo_normalized, nest_normalized, atol=1e-4)
    
    # Check if featurizer has forward_inputs (Log Mel before normalization)
    print(f"\n3. Checking for forward_inputs (Log Mel before normalization):")
    nemo_inputs = nemo_feat.get('forward_inputs', [])
    nest_inputs = nest_feat.get('forward_inputs', [])
    
    if nemo_inputs and nest_inputs:
        print(f"   NeMo inputs: {len(nemo_inputs)} tensors")
        for i, inp in enumerate(nemo_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"     Input[{i}]: shape={inp.shape}, dtype={inp.dtype}")
        
        print(f"   nest inputs: {len(nest_inputs)} tensors")
        for i, inp in enumerate(nest_inputs):
            if isinstance(inp, torch.Tensor):
                print(f"     Input[{i}]: shape={inp.shape}, dtype={inp.dtype}")
        
        # Compare inputs
        if len(nemo_inputs) >= 1 and len(nest_inputs) >= 1:
            nemo_audio = nemo_inputs[0]
            nest_audio = nest_inputs[0]
            print(f"\n4. Comparing Featurizer Inputs (audio):")
            compare_tensors("Audio input", nemo_audio, nest_audio, atol=1e-6)
    else:
        print("   No forward_inputs found")
    
    # Try to find intermediate layers that might have Log Mel
    print(f"\n5. Looking for intermediate layers with Log Mel:")
    log_mel_candidates = []
    for layer_name in nemo_layers.keys():
        if 'log' in layer_name.lower() or 'mel' in layer_name.lower() or 'stft' in layer_name.lower():
            log_mel_candidates.append(layer_name)
    
    if log_mel_candidates:
        print(f"   Found candidates: {log_mel_candidates}")
    else:
        print("   No Log Mel candidates found")
    
    # Analyze the mismatch pattern
    print(f"\n6. Analyzing Mismatch Pattern:")
    # Get first sample
    nemo_sample = nemo_normalized[0]  # [D, T]
    nest_sample = nest_normalized[0]
    
    # Check if values are shifted/permuted
    nemo_flat = nemo_sample.flatten()
    nest_flat = nest_sample.flatten()
    
    # Sort values
    nemo_sorted = torch.sort(nemo_flat)[0]
    nest_sorted = torch.sort(nest_flat)[0]
    
    # Compare sorted values
    sorted_diff = (nemo_sorted - nest_sorted).abs().max().item()
    print(f"   Max diff after sorting: {sorted_diff:.6e}")
    
    if sorted_diff < 1e-3:
        print("   [INFO] Values are similar but in different positions - possible time shift or permutation")
    else:
        print("   [INFO] Values themselves are different - not just a permutation")
    
    # Check valid region only
    valid_len_nemo = nemo_seq_len[0].item()
    valid_len_nest = nest_seq_len[0].item()
    
    print(f"\n7. Valid Region Analysis:")
    print(f"   NeMo valid_len: {valid_len_nemo}")
    print(f"   nest valid_len: {valid_len_nest}")
    
    if valid_len_nemo == valid_len_nest:
        nemo_valid = nemo_sample[:, :valid_len_nemo]
        nest_valid = nest_sample[:, :valid_len_nest]
        
        print(f"\n8. Comparing Valid Regions Only:")
        compare_tensors("Valid region", nemo_valid, nest_valid, atol=1e-4)
        
        # Check stats
        print(f"\n   NeMo valid region stats:")
        print(f"     mean: {nemo_valid.mean().item():.6f}")
        print(f"     std: {nemo_valid.std().item():.6f}")
        print(f"     min: {nemo_valid.min().item():.6f}")
        print(f"     max: {nemo_valid.max().item():.6f}")
        
        print(f"\n   nest valid region stats:")
        print(f"     mean: {nest_valid.mean().item():.6f}")
        print(f"     std: {nest_valid.std().item():.6f}")
        print(f"     min: {nest_valid.min().item():.6f}")
        print(f"     max: {nest_valid.max().item():.6f}")
    
    print("\n" + "="*80)
    print("Diagnosis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

