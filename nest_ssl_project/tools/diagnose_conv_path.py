#!/usr/bin/env python3
"""
Diagnose which code path is taken in ConvSubsampling.forward
"""

import torch
import sys
sys.path.insert(0, '.')

from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_dir", type=str, required=True)
    parser.add_argument("--nest_dir", type=str, required=True)
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir)
    nest_dir = Path(args.nest_dir)
    
    print("="*80)
    print("CONV PATH DIAGNOSIS")
    print("="*80)
    
    # Load batch data
    step_dir = nemo_dir / "step_1"
    if not step_dir.exists():
        step_dir = nemo_dir / "step_0"
    
    batch_path = step_dir / "batch.pt"
    batch = torch.load(batch_path, map_location='cpu', weights_only=False)
    
    print(f"\nBatch audio shape: {batch['audio'].shape}")
    print(f"Batch audio_len: {batch['audio_len']}")
    
    # Create a simple test input similar to preprocessor output
    # preprocessor output shape is [2, 80, 1584]
    # After transpose in encoder: [2, 1584, 80]
    test_input = torch.randn(2, 1584, 80)
    test_lengths = torch.tensor([1249, 1582])
    
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test lengths: {test_lengths}")
    
    # Test NeMo's ConvSubsampling
    print("\n" + "="*40)
    print("Testing NeMo ConvSubsampling")
    print("="*40)
    
    try:
        from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling as NeMoConvSubsampling
        
        nemo_conv = NeMoConvSubsampling(
            subsampling='dw_striding',
            subsampling_factor=8,
            feat_in=80,
            feat_out=512,
            conv_channels=256,
            subsampling_conv_chunking_factor=1,
        )
        
        print(f"NeMo conv2d_subsampling: {nemo_conv.conv2d_subsampling}")
        print(f"NeMo subsampling_conv_chunking_factor: {nemo_conv.subsampling_conv_chunking_factor}")
        
        # Check which path will be taken
        x_ceil = 2**31 / nemo_conv._conv_channels * nemo_conv._stride * nemo_conv._stride
        need_to_split = torch.numel(test_input) > x_ceil
        print(f"NeMo x_ceil: {x_ceil}")
        print(f"NeMo test_input numel: {torch.numel(test_input)}")
        print(f"NeMo need_to_split: {need_to_split}")
        
        # Run forward
        nemo_out, nemo_out_len = nemo_conv(test_input, test_lengths)
        print(f"NeMo output shape: {nemo_out.shape}")
        print(f"NeMo output lengths: {nemo_out_len}")
        
    except Exception as e:
        print(f"NeMo error: {e}")
    
    # Test nest's ConvSubsampling
    print("\n" + "="*40)
    print("Testing nest ConvSubsampling")
    print("="*40)
    
    try:
        from modules.conformer_encoder import ConvSubsampling as NestConvSubsampling
        
        nest_conv = NestConvSubsampling(
            subsampling='dw_striding',
            subsampling_factor=8,
            feat_in=80,
            feat_out=512,
            conv_channels=256,
            subsampling_conv_chunking_factor=1,
        )
        
        print(f"nest conv2d_subsampling: {nest_conv.conv2d_subsampling}")
        print(f"nest subsampling_conv_chunking_factor: {nest_conv.subsampling_conv_chunking_factor}")
        
        # Check which path will be taken
        x_ceil = 2**31 / nest_conv._conv_channels * nest_conv._stride * nest_conv._stride
        need_to_split = torch.numel(test_input) > x_ceil
        print(f"nest x_ceil: {x_ceil}")
        print(f"nest test_input numel: {torch.numel(test_input)}")
        print(f"nest need_to_split: {need_to_split}")
        
        # Run forward
        nest_out, nest_out_len = nest_conv(test_input, test_lengths)
        print(f"nest output shape: {nest_out.shape}")
        print(f"nest output lengths: {nest_out_len}")
        
    except Exception as e:
        print(f"nest error: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare if both succeeded
    print("\n" + "="*40)
    print("Comparison")
    print("="*40)
    
    try:
        diff = (nemo_out - nest_out).abs()
        print(f"Max diff: {diff.max().item():.6e}")
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
    except Exception as e:
        print(f"Comparison error: {e}")


if __name__ == '__main__':
    main()

