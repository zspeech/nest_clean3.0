#!/usr/bin/env python3
"""
Test normalize_batch function directly to see if there are any differences.
"""

import torch
import sys
from pathlib import Path

# Add NeMo to path
nemo_path = Path(__file__).parent.parent.parent / "NeMo"
if nemo_path.exists():
    sys.path.insert(0, str(nemo_path))

# Import normalize_batch from both implementations
sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.audio_preprocessing import normalize_batch as nest_normalize_batch

try:
    from nemo.collections.asr.parts.preprocessing.features import normalize_batch as nemo_normalize_batch
except ImportError:
    print("Warning: Could not import NeMo's normalize_batch")
    nemo_normalize_batch = None

def test_normalize_batch():
    """Test normalize_batch with sample data."""
    print("="*80)
    print("Testing normalize_batch function")
    print("="*80)
    
    # Create test data
    torch.manual_seed(42)
    batch_size = 2
    n_features = 80
    max_time = 100
    
    test_x = torch.randn(batch_size, n_features, max_time)
    test_seq_len = torch.tensor([80, 100], dtype=torch.long)
    
    print(f"\nInput shape: {test_x.shape}")
    print(f"seq_len: {test_seq_len}")
    
    # Test nest implementation
    print("\n1. Testing nest implementation:")
    nest_result, nest_mean, nest_std = nest_normalize_batch(test_x.clone(), test_seq_len.clone(), "per_feature")
    print(f"   Output shape: {nest_result.shape}")
    print(f"   Mean shape: {nest_mean.shape}")
    print(f"   Std shape: {nest_std.shape}")
    print(f"   Output stats: min={nest_result.min().item():.6f}, max={nest_result.max().item():.6f}, mean={nest_result.mean().item():.6f}")
    
    # Test NeMo implementation if available
    if nemo_normalize_batch is not None:
        print("\n2. Testing NeMo implementation:")
        nemo_result, nemo_mean, nemo_std = nemo_normalize_batch(test_x.clone(), test_seq_len.clone(), "per_feature")
        print(f"   Output shape: {nemo_result.shape}")
        print(f"   Mean shape: {nemo_mean.shape}")
        print(f"   Std shape: {nemo_std.shape}")
        print(f"   Output stats: min={nemo_result.min().item():.6f}, max={nemo_result.max().item():.6f}, mean={nemo_result.mean().item():.6f}")
        
        # Compare results
        print("\n3. Comparing results:")
        diff = (nest_result - nemo_result).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        is_close = torch.allclose(nest_result, nemo_result, atol=1e-5, rtol=1e-5)
        print(f"   Match: {is_close}")
        print(f"   Max diff: {max_diff:.6e}")
        print(f"   Mean diff: {mean_diff:.6e}")
        
        if not is_close:
            max_idx = diff.argmax()
            max_idx_unraveled = torch.unravel_index(max_idx, nest_result.shape)
            print(f"   Max diff at index: {max_idx_unraveled}")
            print(f"   nest value: {nest_result.flatten()[max_idx].item():.6f}")
            print(f"   NeMo value: {nemo_result.flatten()[max_idx].item():.6f}")
        
        # Compare mean and std
        print("\n4. Comparing mean and std:")
        mean_diff = (nest_mean - nemo_mean).abs().max().item()
        std_diff = (nest_std - nemo_std).abs().max().item()
        print(f"   Mean diff: {mean_diff:.6e}")
        print(f"   Std diff: {std_diff:.6e}")
    else:
        print("\n2. NeMo implementation not available for comparison")
    
    print("\n" + "="*80)
    print("Test complete.")
    print("="*80)


if __name__ == '__main__':
    test_normalize_batch()

