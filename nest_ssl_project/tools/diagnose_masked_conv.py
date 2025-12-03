#!/usr/bin/env python3
"""
Diagnose MaskedConvSequential step by step
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')


def main():
    torch.manual_seed(42)
    
    print("="*80)
    print("MASKED CONV SEQUENTIAL DIAGNOSIS")
    print("="*80)
    
    # Create test input - same as what ConvSubsampling receives
    # Input to MaskedConvSequential is (batch, time, features)
    test_input = torch.randn(2, 1584, 80)
    test_lengths = torch.tensor([1249, 1582])
    
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test lengths: {test_lengths}")
    
    # Import both implementations
    from nemo.collections.asr.parts.submodules.subsampling import (
        MaskedConvSequential as NeMoMaskedConvSequential,
        apply_channel_mask as nemo_apply_channel_mask,
        calculate_conv_output_size as nemo_calc_conv_output_size,
    )
    from modules.conformer_encoder import (
        MaskedConvSequential as NestMaskedConvSequential,
        apply_channel_mask as nest_apply_channel_mask,
        calculate_conv_output_size as nest_calc_conv_output_size,
    )
    
    # Create simple conv layers (same as dw_striding)
    def create_layers():
        layers = []
        # Layer 0: Conv2d with stride
        layers.append(nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1))
        # Layer 1: ReLU
        layers.append(nn.ReLU())
        # Layer 2: Depthwise Conv2d with stride
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256))
        # Layer 3: Pointwise Conv2d
        layers.append(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        # Layer 4: ReLU
        layers.append(nn.ReLU())
        # Layer 5: Depthwise Conv2d with stride
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256))
        # Layer 6: Pointwise Conv2d
        layers.append(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        return layers
    
    torch.manual_seed(42)
    nemo_layers = create_layers()
    torch.manual_seed(42)
    nest_layers = create_layers()
    
    # Create MaskedConvSequential
    nemo_conv = NeMoMaskedConvSequential(*nemo_layers)
    nest_conv = NestMaskedConvSequential(*nest_layers)
    
    # Sync weights
    nest_conv.load_state_dict(nemo_conv.state_dict())
    
    print("\n" + "="*40)
    print("Step-by-step comparison")
    print("="*40)
    
    # Manually trace through the forward pass
    x_nemo = test_input.clone()
    x_nest = test_input.clone()
    lengths_nemo = test_lengths.clone().float()
    lengths_nest = test_lengths.clone().float()
    
    # Step 1: unsqueeze
    x_nemo = x_nemo.unsqueeze(1)
    x_nest = x_nest.unsqueeze(1)
    print(f"\nAfter unsqueeze: NeMo={x_nemo.shape}, nest={x_nest.shape}")
    print(f"  Max diff: {(x_nemo - x_nest).abs().max().item():.6e}")
    
    # Step 2: create initial mask
    def nemo_create_mask(tensor, lengths):
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)
    
    def nest_create_mask(tensor, lengths):
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device, dtype=torch.long).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)
    
    mask_nemo = nemo_create_mask(x_nemo, lengths_nemo.long())
    mask_nest = nest_create_mask(x_nest, lengths_nest.long())
    print(f"\nInitial mask: NeMo={mask_nemo.shape}, nest={mask_nest.shape}")
    print(f"  Max diff: {(mask_nemo - mask_nest).abs().max().item():.6e}")
    print(f"  NeMo mask sum: {mask_nemo.sum().item()}, nest mask sum: {mask_nest.sum().item()}")
    
    # Process each layer
    for i, (nemo_layer, nest_layer) in enumerate(zip(nemo_layers, nest_layers)):
        print(f"\n--- Layer {i}: {type(nemo_layer).__name__} ---")
        
        # Apply mask before layer
        x_nemo_masked = nemo_apply_channel_mask(x_nemo, mask_nemo)
        x_nest_masked = nest_apply_channel_mask(x_nest, mask_nest)
        
        print(f"  After apply_channel_mask:")
        print(f"    NeMo: {x_nemo_masked.shape}, zeros={((x_nemo_masked == 0).sum().item())}")
        print(f"    nest: {x_nest_masked.shape}, zeros={((x_nest_masked == 0).sum().item())}")
        diff = (x_nemo_masked - x_nest_masked).abs()
        print(f"    Max diff: {diff.max().item():.6e}")
        
        if diff.max().item() > 1e-6:
            # Find where they differ
            max_idx = diff.argmax()
            idx = []
            flat_idx = max_idx.item()
            for dim in reversed(x_nemo_masked.shape):
                idx.insert(0, flat_idx % dim)
                flat_idx //= dim
            print(f"    Max diff at {tuple(idx)}: NeMo={x_nemo_masked[tuple(idx)].item():.6f}, nest={x_nest_masked[tuple(idx)].item():.6f}")
            
            # Check mask at that position
            mask_idx = (idx[0], idx[2], idx[3])  # batch, time, features
            print(f"    Mask at {mask_idx}: NeMo={mask_nemo[mask_idx].item()}, nest={mask_nest[mask_idx].item()}")
        
        # Apply layer
        x_nemo = nemo_layer(x_nemo_masked)
        x_nest = nest_layer(x_nest_masked)
        
        print(f"  After layer:")
        print(f"    NeMo: {x_nemo.shape}")
        print(f"    nest: {x_nest.shape}")
        diff = (x_nemo - x_nest).abs()
        print(f"    Max diff: {diff.max().item():.6e}")
        
        # Update lengths and mask for stride operations
        if hasattr(nemo_layer, 'stride') and nemo_layer.stride != (1, 1):
            padding = nemo_layer.padding
            
            lengths_nemo = nemo_calc_conv_output_size(
                lengths_nemo, nemo_layer.kernel_size[0], nemo_layer.stride[0], padding
            )
            lengths_nest = nest_calc_conv_output_size(
                lengths_nest, nest_layer.kernel_size[0], nest_layer.stride[0], padding
            )
            
            print(f"  Updated lengths: NeMo={lengths_nemo}, nest={lengths_nest}")
            
            mask_nemo = nemo_create_mask(x_nemo, lengths_nemo.long())
            mask_nest = nest_create_mask(x_nest, lengths_nest.long())
            
            print(f"  Updated mask: NeMo sum={mask_nemo.sum().item()}, nest sum={mask_nest.sum().item()}")
            print(f"  Mask diff: {(mask_nemo - mask_nest).abs().max().item():.6e}")
    
    # Final masking
    print(f"\n--- Final masking ---")
    x_nemo_final = nemo_apply_channel_mask(x_nemo, mask_nemo)
    x_nest_final = nest_apply_channel_mask(x_nest, mask_nest)
    
    print(f"  NeMo: {x_nemo_final.shape}, zeros={((x_nemo_final == 0).sum().item())}")
    print(f"  nest: {x_nest_final.shape}, zeros={((x_nest_final == 0).sum().item())}")
    diff = (x_nemo_final - x_nest_final).abs()
    print(f"  Max diff: {diff.max().item():.6e}")


if __name__ == '__main__':
    main()

