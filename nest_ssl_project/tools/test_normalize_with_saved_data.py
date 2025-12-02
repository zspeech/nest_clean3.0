#!/usr/bin/env python3
"""
Test normalize_batch with actual saved data from training runs.
This extracts the Log Mel input (before normalization) and manually calls normalize_batch.
"""

import torch
import pickle
import sys
from pathlib import Path
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.audio_preprocessing import normalize_batch

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
        print(f"    t1 value: {t1.flatten()[max_idx].item():.6f}")
        print(f"    t2 value: {t2.flatten()[max_idx].item():.6f}")
        print(f"    t1 stats: min={t1.min().item():.6f}, max={t1.max().item():.6f}, mean={t1.float().mean().item():.6f}")
        print(f"    t2 stats: min={t2.min().item():.6f}, max={t2.max().item():.6f}, mean={t2.float().mean().item():.6f}")
    
    return is_close


def main():
    parser = argparse.ArgumentParser(description="Test normalize_batch with saved data")
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
    print("TESTING normalize_batch WITH SAVED DATA")
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
    
    # Now we need to reverse-normalize to get the Log Mel input
    # But we don't have mean and std... So we need to compute them from the STFT diagnosis
    
    print("\n2. We need to extract Log Mel from STFT diagnosis...")
    print("   Please run diagnose_featurizer_stft.py first to get Log Mel values")
    print("   Then we can manually call normalize_batch and compare")
    
    # Actually, let's try to load from STFT diagnosis if it exists
    # Or we can compute Log Mel from the batch data
    
    print("\n3. Attempting to compute Log Mel from batch data...")
    print("   This requires loading the batch and running STFT manually")
    print("   For now, let's check if we can extract it from the saved debug inputs")
    
    # Check if debug inputs were saved
    if hasattr(normalize_batch, '_debug_inputs'):
        print("\n4. Found debug inputs from normalize_batch:")
        debug = normalize_batch._debug_inputs
        print(f"   x shape: {debug['x_shape']}")
        print(f"   seq_len values: {debug['seq_len_values']}")
        
        # Manually call normalize_batch with these inputs
        print("\n5. Manually calling normalize_batch with debug inputs:")
        result, mean, std = normalize_batch(debug['x'], debug['seq_len'], "per_feature")
        print(f"   Output shape: {result.shape}")
        print(f"   Output stats: min={result.min().item():.6f}, max={result.max().item():.6f}, mean={result.float().mean().item():.6f}")
        
        # Compare with actual nest output
        print("\n6. Comparing manual output with actual nest output:")
        compare_tensors("Manual vs Actual", result, nest_final, atol=1e-4)
    else:
        print("\n4. No debug inputs found. Please run training first to capture them.")
    
    print("\n" + "="*80)
    print("Test complete.")
    print("="*80)


if __name__ == '__main__':
    main()

