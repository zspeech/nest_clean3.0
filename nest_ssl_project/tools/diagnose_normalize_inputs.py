#!/usr/bin/env python3
"""
Diagnose normalize_batch inputs - compare what's actually passed to normalize_batch.
"""

import torch
import pickle
import sys
from pathlib import Path
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    
    if t1.dtype != t2.dtype:
        print(f"  {name}: Dtype mismatch: {t1.dtype} vs {t2.dtype}")
    
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
        print(f"    NeMo value: {t1.flatten()[max_idx].item():.6f}")
        print(f"    nest value: {t2.flatten()[max_idx].item():.6f}")
        print(f"    NeMo stats: min={t1.min().item():.6f}, max={t1.max().item():.6f}, mean={t1.float().mean().item():.6f}")
        print(f"    nest stats: min={t2.min().item():.6f}, max={t2.max().item():.6f}, mean={t2.float().mean().item():.6f}")
    
    return is_close


def main():
    parser = argparse.ArgumentParser(description="Diagnose normalize_batch inputs")
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
    print("NORMALIZE_BATCH INPUTS DIAGNOSIS")
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
        print("Featurizer layer not found")
        return
    
    nemo_feat = nemo_layers['preprocessor.featurizer']
    nest_feat = nest_layers['preprocessor.featurizer']
    
    nemo_outputs = nemo_feat.get('forward_outputs', [])
    nest_outputs = nest_feat.get('forward_outputs', [])
    
    if not nemo_outputs or not nest_outputs:
        print("No outputs found")
        return
    
    nemo_final = nemo_outputs[0]  # [B, D, T] - after normalization
    nest_final = nest_outputs[0]
    nemo_seq_len = nemo_outputs[1]
    nest_seq_len = nest_outputs[1]
    
    print("\n1. Final Outputs (after normalization):")
    compare_tensors("Final output", nemo_final, nest_final, atol=1e-4)
    
    # Reconstruct normalization inputs
    # We need to reverse-normalize to get the log-mel input
    print("\n2. Reconstructing Normalization Inputs (Log Mel):")
    
    from modules.audio_preprocessing import normalize_batch
    
    # Try to reverse-normalize
    # If we know the mean and std, we can reverse: x_norm = (x - mean) / std
    # So: x = x_norm * std + mean
    
    # But we don't have mean and std... Let's try a different approach
    # We can simulate normalization on the final output to see if we can match
    
    # Actually, let's check if we can extract log-mel from the STFT diagnosis
    # Or we can manually compute log-mel from the audio
    
    print("   This requires manual STFT computation...")
    print("   Please run diagnose_featurizer_stft.py to get Log Mel values")
    
    # Alternative: Check if seq_len values match
    print("\n3. Checking seq_len values:")
    compare_tensors("seq_len", nemo_seq_len, nest_seq_len)
    
    # Check if the issue is in the mask generation
    print("\n4. Checking mask generation logic:")
    batch_size = nemo_final.shape[0]
    max_time = nemo_final.shape[2]
    
    # Generate masks using the same logic as normalize_batch
    nemo_mask = torch.arange(max_time, device=nemo_final.device).unsqueeze(0).expand(batch_size, max_time) < nemo_seq_len.unsqueeze(1)
    nest_mask = torch.arange(max_time, device=nest_final.device).unsqueeze(0).expand(batch_size, max_time) < nest_seq_len.unsqueeze(1)
    
    print(f"   Mask shapes: NeMo {nemo_mask.shape}, nest {nest_mask.shape}")
    print(f"   Mask match: {torch.equal(nemo_mask, nest_mask)}")
    
    if not torch.equal(nemo_mask, nest_mask):
        diff_mask = (nemo_mask != nest_mask).sum().item()
        print(f"   Mask differences: {diff_mask} positions")
        # Find where they differ
        diff_positions = torch.nonzero(nemo_mask != nest_mask)
        print(f"   First few diff positions: {diff_positions[:5]}")
    
    print("\n" + "="*80)
    print("Diagnosis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

