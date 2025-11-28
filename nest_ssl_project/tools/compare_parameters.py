#!/usr/bin/env python3
"""
Script to compare parameters between our ConformerEncoder and NeMo's ConformerEncoder.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    
    print("="*80)
    print("Comparing ConformerEncoder Parameters")
    print("="*80)
    
    # Try to import NeMo's ConformerEncoder
    try:
        from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder as NeMoConformerEncoder
        HAVE_NEMO = True
        print("✓ Successfully imported NeMo's ConformerEncoder")
    except ImportError as e:
        print(f"✗ Could not import NeMo's ConformerEncoder: {e}")
        HAVE_NEMO = False
    
    # Import our ConformerEncoder
    try:
        from modules.conformer_encoder import ConformerEncoder as OurConformerEncoder
        HAVE_OURS = True
        print("✓ Successfully imported our ConformerEncoder")
    except ImportError as e:
        print(f"✗ Could not import our ConformerEncoder: {e}")
        HAVE_OURS = False
    
    if not HAVE_OURS:
        print("\nCannot proceed without our ConformerEncoder")
        sys.exit(1)
    
    # Configuration for Large (120M) model
    config = {
        'feat_in': 80,
        'n_layers': 17,
        'd_model': 512,
        'n_heads': 8,
        'conv_kernel_size': 9,
        'subsampling_factor': 8,
        'subsampling_conv_channels': 256,
        'self_attention_model': 'rel_pos',
        'xscaling': True,
        'untie_biases': True,
        'use_bias': True,
    }
    
    print("\n" + "="*80)
    print("Creating models with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*80)
    
    # Create our model
    print("\nCreating our ConformerEncoder...")
    try:
        our_model = OurConformerEncoder(**config)
        our_total = sum(p.numel() for p in our_model.parameters())
        our_trainable = sum(p.numel() for p in our_model.parameters() if p.requires_grad)
        print(f"✓ Our model created successfully")
        print(f"  Total parameters: {our_total:,} ({our_total/1e6:.2f}M)")
        print(f"  Trainable parameters: {our_trainable:,} ({our_trainable/1e6:.2f}M)")
    except Exception as e:
        print(f"✗ Failed to create our model: {e}")
        import traceback
        traceback.print_exc()
        our_model = None
        our_total = 0
    
    # Create NeMo model if available
    if HAVE_NEMO:
        print("\nCreating NeMo's ConformerEncoder...")
        try:
            nemo_model = NeMoConformerEncoder(**config)
            nemo_total = sum(p.numel() for p in nemo_model.parameters())
            nemo_trainable = sum(p.numel() for p in nemo_model.parameters() if p.requires_grad)
            print(f"✓ NeMo model created successfully")
            print(f"  Total parameters: {nemo_total:,} ({nemo_total/1e6:.2f}M)")
            print(f"  Trainable parameters: {nemo_trainable:,} ({nemo_trainable/1e6:.2f}M)")
            
            if our_model:
                diff = nemo_total - our_total
                print(f"\n  Difference: {diff:,} ({diff/1e6:.2f}M)")
                if abs(diff) < 1000:
                    print("  ✓ Parameters match!")
                else:
                    print(f"  ✗ Parameters differ by {abs(diff):,}")
        except Exception as e:
            print(f"✗ Failed to create NeMo model: {e}")
            import traceback
            traceback.print_exc()
            nemo_model = None
            nemo_total = 0
    
    # Detailed breakdown of our model
    if our_model:
        print("\n" + "="*80)
        print("Detailed breakdown of our ConformerEncoder:")
        print("="*80)
        
        def count_module_params(module, prefix=""):
            total = 0
            for name, child in module.named_children():
                child_params = sum(p.numel() for p in child.parameters())
                if child_params > 0:
                    print(f"  {prefix}{name}: {child_params:,} ({child_params/1e6:.2f}M)")
                    total += child_params
            return total
        
        pre_encode_params = sum(p.numel() for p in our_model.pre_encode.parameters())
        pos_enc_params = sum(p.numel() for p in our_model.pos_enc.parameters()) if our_model.pos_enc else 0
        layers_params = sum(p.numel() for layer in our_model.layers for p in layer.parameters())
        output_proj_params = sum(p.numel() for p in our_model.output_proj.parameters())
        
        print(f"\nPre-encoder (subsampling): {pre_encode_params:,} ({pre_encode_params/1e6:.2f}M)")
        if our_model.pos_enc:
            print(f"Positional encoding:        {pos_enc_params:,} ({pos_enc_params/1e6:.2f}M)")
        print(f"Conformer layers (17):     {layers_params:,} ({layers_params/1e6:.2f}M)")
        print(f"Output projection:         {output_proj_params:,} ({output_proj_params/1e6:.2f}M)")
        print(f"Total:                     {our_total:,} ({our_total/1e6:.2f}M)")
        
        # Per-layer breakdown
        if len(our_model.layers) > 0:
            print(f"\nPer-layer breakdown (first layer):")
            for name, param in our_model.layers[0].named_parameters():
                print(f"  {name:60s} {param.numel():>10,}")
    
    print("\n" + "="*80)
    print("Recommendation:")
    print("="*80)
    if HAVE_NEMO and abs(nemo_total - our_total) < 1000:
        print("✓ Our implementation matches NeMo! No need to use NeMo's code.")
    elif HAVE_NEMO:
        print("⚠ Parameters differ. You can:")
        print("  1. Use NeMo's ConformerEncoder directly (remove remapping in serialization.py)")
        print("  2. Continue fixing our implementation to match NeMo")
    else:
        print("⚠ Cannot compare with NeMo. Please install NeMo to compare.")
    print("="*80)

if __name__ == "__main__":
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. Please install PyTorch first.")
        sys.exit(1)

