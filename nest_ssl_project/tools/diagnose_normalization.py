#!/usr/bin/env python3
"""
Diagnose normalization step differences.
Compare inputs and outputs of normalize_batch between NeMo and nest.
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
    
    if t1.dtype != t2.dtype:
        print(f"  {name}: Dtype mismatch: {t1.dtype} vs {t2.dtype}")
    
    # Convert to float for comparison
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
    parser = argparse.ArgumentParser(description="Diagnose normalization differences")
    parser.add_argument("--nemo_dir", type=str, required=True, help="NeMo output directory")
    parser.add_argument("--nest_dir", type=str, required=True, help="nest output directory")
    parser.add_argument("--step", type=int, default=None, help="Step to compare (default: auto-detect)")
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir)
    nest_dir = Path(args.nest_dir)
    
    # Auto-detect step if not provided
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
    
    print(f"Analyzing step: {args.step}")
    
    # Load layer outputs
    try:
        with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
            nemo_layers = pickle.load(f)
        with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
            nest_layers = pickle.load(f)
    except Exception as e:
        print(f"Error loading layer outputs: {e}")
        return
    
    print("="*80)
    print("NORMALIZATION DIAGNOSIS")
    print("="*80)
    
    # Get featurizer outputs
    if 'preprocessor.featurizer' not in nemo_layers or 'preprocessor.featurizer' not in nest_layers:
        print("Featurizer layer not found in outputs")
        return
    
    nemo_feat = nemo_layers['preprocessor.featurizer']
    nest_feat = nest_layers['preprocessor.featurizer']
    
    nemo_outputs = nemo_feat.get('forward_outputs', [])
    nest_outputs = nest_feat.get('forward_outputs', [])
    
    if not nemo_outputs or not nest_outputs:
        print("No outputs found")
        return
    
    nemo_final = nemo_outputs[0]  # [B, D, T]
    nest_final = nest_outputs[0]  # [B, D, T]
    nemo_seq_len = nemo_outputs[1]
    nest_seq_len = nest_outputs[1]
    
    print("\n1. Final Featurizer Outputs (after normalization + mask + pad):")
    compare_tensors("Final output", nemo_final, nest_final, atol=1e-4)
    compare_tensors("seq_len", nemo_seq_len, nest_seq_len)
    
    # Simulate normalization manually
    print("\n2. Simulating Normalization:")
    
    # We need to reconstruct the log-mel input to normalization
    # This is tricky - we'd need to reverse the normalization
    # Instead, let's check if we can extract intermediate values
    
    # Check if there are any intermediate layer outputs
    print("\n3. Checking for intermediate layer outputs...")
    print(f"  Available NeMo layers: {sorted(nemo_layers.keys())[:10]}...")
    print(f"  Available nest layers: {sorted(nest_layers.keys())[:10]}...")
    
    # Try to find preprocessor output (before featurizer)
    if 'preprocessor' in nemo_layers and 'preprocessor' in nest_layers:
        nemo_prep = nemo_layers['preprocessor']
        nest_prep = nest_layers['preprocessor']
        
        nemo_prep_outputs = nemo_prep.get('forward_outputs', [])
        nest_prep_outputs = nest_prep.get('forward_outputs', [])
        
        if nemo_prep_outputs and nest_prep_outputs:
            print("\n4. Preprocessor Outputs (should match featurizer inputs):")
            compare_tensors("preprocessor output[0]", nemo_prep_outputs[0], nest_prep_outputs[0], atol=1e-4)
            compare_tensors("preprocessor output[1]", nemo_prep_outputs[1], nest_prep_outputs[1])
    
    # Manual normalization test
    print("\n5. Manual Normalization Test:")
    print("  Testing normalize_batch function with sample data...")
    
    # Import normalize_batch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from modules.audio_preprocessing import normalize_batch
    
    # Create test data
    batch_size = 2
    n_features = 80
    max_time = 100
    
    # Test with random data
    torch.manual_seed(42)
    test_x = torch.randn(batch_size, n_features, max_time)
    test_seq_len = torch.tensor([80, 100], dtype=torch.long)
    
    # Test normalization
    norm_x, x_mean, x_std = normalize_batch(test_x, test_seq_len, "per_feature")
    
    print(f"  Test normalization successful")
    print(f"  Output shape: {norm_x.shape}")
    print(f"  Mean shape: {x_mean.shape}")
    print(f"  Std shape: {x_std.shape}")
    
    # Check if normalization worked (mean should be ~0 for valid regions)
    for i in range(batch_size):
        valid_region = norm_x[i, :, :test_seq_len[i]]
        mean_val = valid_region.mean().item()
        print(f"  Sample {i} valid region mean: {mean_val:.6f} (should be ~0)")
    
    print("\n" + "="*80)
    print("Diagnosis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

