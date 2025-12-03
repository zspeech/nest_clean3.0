#!/usr/bin/env python3
"""
Diagnose ConvSubsampling layer by layer
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from pathlib import Path
import argparse


def compare_modules(name, nemo_module, nest_module):
    """Compare two modules' parameters."""
    print(f"\n--- {name} ---")
    
    nemo_params = dict(nemo_module.named_parameters())
    nest_params = dict(nest_module.named_parameters())
    
    all_keys = set(nemo_params.keys()) | set(nest_params.keys())
    
    for key in sorted(all_keys):
        if key in nemo_params and key in nest_params:
            nemo_p = nemo_params[key]
            nest_p = nest_params[key]
            if nemo_p.shape == nest_p.shape:
                diff = (nemo_p - nest_p).abs().max().item()
                status = "[OK]" if diff < 1e-6 else "[DIFF]"
                print(f"  {status} {key}: shape={nemo_p.shape}, max_diff={diff:.6e}")
            else:
                print(f"  [SHAPE] {key}: NeMo={nemo_p.shape}, nest={nest_p.shape}")
        elif key in nemo_params:
            print(f"  [MISSING in nest] {key}")
        else:
            print(f"  [MISSING in NeMo] {key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    print("="*80)
    print("CONV LAYER-BY-LAYER DIAGNOSIS")
    print("="*80)
    
    # Create test input
    test_input = torch.randn(2, 1584, 80)
    test_lengths = torch.tensor([1249, 1582])
    
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test lengths: {test_lengths}")
    
    # Create NeMo ConvSubsampling
    print("\n" + "="*40)
    print("Creating modules")
    print("="*40)
    
    from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling as NeMoConvSubsampling
    from modules.conformer_encoder import ConvSubsampling as NestConvSubsampling
    
    # Use same seed for initialization
    torch.manual_seed(args.seed)
    nemo_conv = NeMoConvSubsampling(
        subsampling='dw_striding',
        subsampling_factor=8,
        feat_in=80,
        feat_out=512,
        conv_channels=256,
        subsampling_conv_chunking_factor=1,
    )
    
    torch.manual_seed(args.seed)
    nest_conv = NestConvSubsampling(
        subsampling='dw_striding',
        subsampling_factor=8,
        feat_in=80,
        feat_out=512,
        conv_channels=256,
        subsampling_conv_chunking_factor=1,
    )
    
    # Compare parameters
    print("\n" + "="*40)
    print("Comparing parameters")
    print("="*40)
    compare_modules("ConvSubsampling", nemo_conv, nest_conv)
    
    # Copy weights from NeMo to nest
    print("\n" + "="*40)
    print("Copying weights from NeMo to nest")
    print("="*40)
    nest_conv.load_state_dict(nemo_conv.state_dict())
    print("Weights copied!")
    
    # Now compare outputs
    print("\n" + "="*40)
    print("Comparing outputs after weight sync")
    print("="*40)
    
    torch.manual_seed(args.seed)
    nemo_out, nemo_len = nemo_conv(test_input.clone(), test_lengths.clone())
    
    torch.manual_seed(args.seed)
    nest_out, nest_len = nest_conv(test_input.clone(), test_lengths.clone())
    
    diff = (nemo_out - nest_out).abs()
    print(f"Max diff: {diff.max().item():.6e}")
    print(f"Mean diff: {diff.mean().item():.6e}")
    
    if diff.max().item() < 1e-5:
        print("[OK] Outputs match after weight sync!")
    else:
        print("[FAIL] Outputs still differ after weight sync!")
        
        # Find max diff location
        max_idx = diff.argmax()
        idx = []
        flat_idx = max_idx.item()
        for dim in reversed(nemo_out.shape):
            idx.insert(0, flat_idx % dim)
            flat_idx //= dim
        print(f"Max diff at {tuple(idx)}: NeMo={nemo_out[tuple(idx)].item():.6f}, nest={nest_out[tuple(idx)].item():.6f}")
    
    # Compare intermediate outputs using hooks
    print("\n" + "="*40)
    print("Comparing intermediate outputs")
    print("="*40)
    
    nemo_intermediates = {}
    nest_intermediates = {}
    
    def make_hook(storage, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[name] = (output[0].detach().clone(), output[1].detach().clone() if len(output) > 1 else None)
            else:
                storage[name] = output.detach().clone()
        return hook
    
    # Register hooks on conv layers
    for name, module in nemo_conv.conv.named_modules():
        if name:
            module.register_forward_hook(make_hook(nemo_intermediates, f"conv.{name}"))
    nemo_conv.conv.register_forward_hook(make_hook(nemo_intermediates, "conv"))
    nemo_conv.out.register_forward_hook(make_hook(nemo_intermediates, "out"))
    
    for name, module in nest_conv.conv.named_modules():
        if name:
            module.register_forward_hook(make_hook(nest_intermediates, f"conv.{name}"))
    nest_conv.conv.register_forward_hook(make_hook(nest_intermediates, "conv"))
    nest_conv.out.register_forward_hook(make_hook(nest_intermediates, "out"))
    
    # Run forward again
    torch.manual_seed(args.seed)
    nemo_out2, _ = nemo_conv(test_input.clone(), test_lengths.clone())
    
    torch.manual_seed(args.seed)
    nest_out2, _ = nest_conv(test_input.clone(), test_lengths.clone())
    
    # Compare intermediates
    all_keys = sorted(set(nemo_intermediates.keys()) | set(nest_intermediates.keys()))
    
    for key in all_keys:
        if key in nemo_intermediates and key in nest_intermediates:
            nemo_val = nemo_intermediates[key]
            nest_val = nest_intermediates[key]
            
            if isinstance(nemo_val, tuple):
                nemo_tensor = nemo_val[0]
                nest_tensor = nest_val[0] if isinstance(nest_val, tuple) else nest_val
            else:
                nemo_tensor = nemo_val
                nest_tensor = nest_val[0] if isinstance(nest_val, tuple) else nest_val
            
            if nemo_tensor.shape != nest_tensor.shape:
                print(f"[SHAPE] {key}: NeMo={nemo_tensor.shape}, nest={nest_tensor.shape}")
            else:
                diff = (nemo_tensor - nest_tensor).abs()
                max_diff = diff.max().item()
                status = "[OK]" if max_diff < 1e-5 else "[FAIL]"
                print(f"{status} {key}: shape={nemo_tensor.shape}, max_diff={max_diff:.6e}")
                
                if max_diff > 1e-5:
                    # Check zeros
                    nemo_zeros = (nemo_tensor == 0).sum().item()
                    nest_zeros = (nest_tensor == 0).sum().item()
                    print(f"       NeMo zeros: {nemo_zeros}, nest zeros: {nest_zeros}")
        elif key in nemo_intermediates:
            print(f"[MISSING in nest] {key}")
        else:
            print(f"[MISSING in NeMo] {key}")


if __name__ == '__main__':
    main()

