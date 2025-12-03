#!/usr/bin/env python3
"""
Directly compare NeMo and nest ConvSubsampling with same weights
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')


def main():
    torch.manual_seed(42)
    
    print("="*80)
    print("DIRECT CONVSUBSAMPLING COMPARISON")
    print("="*80)
    
    # Create test input
    test_input = torch.randn(2, 1584, 80)
    test_lengths = torch.tensor([1249, 1582])
    
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test lengths: {test_lengths}")
    
    # Import both implementations
    from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling as NeMoConvSubsampling
    from modules.conformer_encoder import ConvSubsampling as NestConvSubsampling
    
    # Create with same seed
    torch.manual_seed(42)
    nemo_conv = NeMoConvSubsampling(
        subsampling='dw_striding',
        subsampling_factor=8,
        feat_in=80,
        feat_out=512,
        conv_channels=256,
        subsampling_conv_chunking_factor=1,
    )
    
    torch.manual_seed(42)
    nest_conv = NestConvSubsampling(
        subsampling='dw_striding',
        subsampling_factor=8,
        feat_in=80,
        feat_out=512,
        conv_channels=256,
        subsampling_conv_chunking_factor=1,
    )
    
    # Check structure
    print("\n" + "="*40)
    print("Module structure")
    print("="*40)
    print(f"NeMo conv type: {type(nemo_conv.conv)}")
    print(f"nest conv type: {type(nest_conv.conv)}")
    print(f"NeMo conv2d_subsampling: {nemo_conv.conv2d_subsampling}")
    print(f"nest conv2d_subsampling: {nest_conv.conv2d_subsampling}")
    
    # Copy weights
    print("\n" + "="*40)
    print("Copying weights from NeMo to nest")
    print("="*40)
    nest_conv.load_state_dict(nemo_conv.state_dict())
    print("Done!")
    
    # Verify weights match
    print("\n" + "="*40)
    print("Verifying weights")
    print("="*40)
    for (n1, p1), (n2, p2) in zip(nemo_conv.named_parameters(), nest_conv.named_parameters()):
        diff = (p1 - p2).abs().max().item()
        status = "[OK]" if diff < 1e-6 else "[FAIL]"
        print(f"{status} {n1}: {diff:.6e}")
    
    # Run forward
    print("\n" + "="*40)
    print("Running forward pass")
    print("="*40)
    
    nemo_out, nemo_len = nemo_conv(test_input.clone(), test_lengths.clone())
    nest_out, nest_len = nest_conv(test_input.clone(), test_lengths.clone())
    
    print(f"NeMo output shape: {nemo_out.shape}")
    print(f"nest output shape: {nest_out.shape}")
    print(f"NeMo lengths: {nemo_len}")
    print(f"nest lengths: {nest_len}")
    
    # Compare outputs
    diff = (nemo_out - nest_out).abs()
    print(f"\nMax diff: {diff.max().item():.6e}")
    print(f"Mean diff: {diff.mean().item():.6e}")
    
    if diff.max().item() < 1e-5:
        print("[OK] Outputs match!")
    else:
        print("[FAIL] Outputs differ!")
        
        # Find max diff location
        max_idx = diff.argmax()
        idx = []
        flat_idx = max_idx.item()
        for dim in reversed(nemo_out.shape):
            idx.insert(0, flat_idx % dim)
            flat_idx //= dim
        print(f"Max diff at {tuple(idx)}: NeMo={nemo_out[tuple(idx)].item():.6f}, nest={nest_out[tuple(idx)].item():.6f}")
    
    # Step-by-step comparison
    print("\n" + "="*40)
    print("Step-by-step comparison")
    print("="*40)
    
    x_nemo = test_input.clone()
    x_nest = test_input.clone()
    
    # Step 1: unsqueeze
    x_nemo = x_nemo.unsqueeze(1)
    x_nest = x_nest.unsqueeze(1)
    print(f"\nAfter unsqueeze:")
    print(f"  NeMo: {x_nemo.shape}")
    print(f"  nest: {x_nest.shape}")
    print(f"  Max diff: {(x_nemo - x_nest).abs().max().item():.6e}")
    
    # Step 2: conv
    x_nemo = nemo_conv.conv(x_nemo)
    x_nest = nest_conv.conv(x_nest)
    print(f"\nAfter conv:")
    print(f"  NeMo: {x_nemo.shape}")
    print(f"  nest: {x_nest.shape}")
    diff = (x_nemo - x_nest).abs()
    print(f"  Max diff: {diff.max().item():.6e}")
    
    if diff.max().item() > 1e-5:
        # Compare layer by layer
        print("\n  Layer-by-layer conv comparison:")
        x_nemo2 = test_input.clone().unsqueeze(1)
        x_nest2 = test_input.clone().unsqueeze(1)
        
        for i, (nemo_layer, nest_layer) in enumerate(zip(nemo_conv.conv, nest_conv.conv)):
            x_nemo2 = nemo_layer(x_nemo2)
            x_nest2 = nest_layer(x_nest2)
            diff = (x_nemo2 - x_nest2).abs()
            status = "[OK]" if diff.max().item() < 1e-5 else "[FAIL]"
            print(f"    {status} Layer {i} ({type(nemo_layer).__name__}): max_diff={diff.max().item():.6e}")
    
    # Step 3: out
    b, c, t, f = x_nemo.size()
    x_nemo = nemo_conv.out(x_nemo.transpose(1, 2).reshape(b, t, -1))
    x_nest = nest_conv.out(x_nest.transpose(1, 2).reshape(b, t, -1))
    print(f"\nAfter out:")
    print(f"  NeMo: {x_nemo.shape}")
    print(f"  nest: {x_nest.shape}")
    print(f"  Max diff: {(x_nemo - x_nest).abs().max().item():.6e}")


if __name__ == '__main__':
    main()

