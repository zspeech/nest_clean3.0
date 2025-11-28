#!/usr/bin/env python3
"""
Script to count model parameters and compare with expected values.
"""

import torch
import torch.nn as nn
from collections import defaultdict

def count_parameters(model, detailed=False):
    """Count parameters in a model."""
    total = 0
    trainable = 0
    param_dict = defaultdict(int)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        if param.requires_grad:
            trainable += num_params
        
        if detailed:
            # Group by module type
            module_type = name.split('.')[0] if '.' in name else name
            param_dict[module_type] += num_params
            print(f"{name:60s} {num_params:>15,}")
    
    if detailed:
        print("\n" + "="*80)
        print("Summary by module type:")
        for module_type, count in sorted(param_dict.items(), key=lambda x: -x[1]):
            print(f"{module_type:60s} {count:>15,} ({count/1e6:.2f}M)")
        print("="*80)
    
    return total, trainable

def main():
    print("Counting ConformerEncoder parameters...")
    print("="*80)
    
    # Import here to avoid issues if modules aren't available
    try:
        from modules.conformer_encoder import ConformerEncoder
        
        # Create model with Large (120M) config
        model = ConformerEncoder(
            feat_in=80,
            n_layers=17,
            d_model=512,
            n_heads=8,
            conv_kernel_size=9,
            subsampling_factor=8,
            subsampling_conv_channels=256,
            self_attention_model='rel_pos',
            xscaling=True,
            untie_biases=True,
            use_bias=True,
        )
        
        print("\nModel configuration:")
        print(f"  d_model: {model.d_model}")
        print(f"  n_layers: {model.n_layers}")
        print(f"  n_heads: 8")
        print(f"  conv_kernel_size: 9")
        print(f"  subsampling_factor: {model.subsampling_factor}")
        print(f"  subsampling_conv_channels: 256")
        print(f"  self_attention_model: rel_pos")
        print(f"  untie_biases: True")
        print(f"  use_bias: True")
        
        print("\n" + "="*80)
        print("Detailed parameter breakdown:")
        print("="*80)
        
        total, trainable = count_parameters(model, detailed=True)
        
        print(f"\nTotal parameters: {total:,} ({total/1e6:.2f}M)")
        print(f"Trainable parameters: {trainable:,} ({trainable/1e6:.2f}M)")
        print(f"Expected: ~120M")
        print(f"Difference: {total/1e6 - 120:.2f}M")
        
        # Count by major components
        print("\n" + "="*80)
        print("Breakdown by major components:")
        print("="*80)
        
        pre_encode_params = sum(p.numel() for p in model.pre_encode.parameters())
        pos_enc_params = sum(p.numel() for p in model.pos_enc.parameters()) if model.pos_enc else 0
        layers_params = sum(p.numel() for layer in model.layers for p in layer.parameters())
        output_proj_params = sum(p.numel() for p in model.output_proj.parameters())
        
        print(f"Pre-encoder (subsampling): {pre_encode_params:,} ({pre_encode_params/1e6:.2f}M)")
        print(f"Positional encoding:        {pos_enc_params:,} ({pos_enc_params/1e6:.2f}M)")
        print(f"Conformer layers (17):     {layers_params:,} ({layers_params/1e6:.2f}M)")
        print(f"Output projection:         {output_proj_params:,} ({output_proj_params/1e6:.2f}M)")
        print(f"Total:                     {total:,} ({total/1e6:.2f}M)")
        
        # Count per layer
        print("\n" + "="*80)
        print("Parameters per ConformerLayer:")
        print("="*80)
        if len(model.layers) > 0:
            layer_params = sum(p.numel() for p in model.layers[0].parameters())
            print(f"Single layer: {layer_params:,} ({layer_params/1e6:.2f}M)")
            print(f"All {len(model.layers)} layers: {layers_params:,} ({layers_params/1e6:.2f}M)")
            
            # Breakdown of a single layer
            print("\nSingle layer breakdown:")
            for name, param in model.layers[0].named_parameters():
                print(f"  {name:50s} {param.numel():>10,}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

