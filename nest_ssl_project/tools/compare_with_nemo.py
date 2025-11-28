#!/usr/bin/env python3
"""
Script to compare our implementation with NeMo's implementation.
Compares:
1. Mask loss calculation
2. Network structure
3. Parameter count
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_structure(model, name="Model", indent=0):
    """Print model structure recursively."""
    prefix = "  " * indent
    print(f"{prefix}{name}:")
    for name, module in model.named_children():
        param_count = count_parameters(module)
        if param_count > 0:
            print(f"{prefix}  {name}: {param_count:,} params")
        if len(list(module.children())) > 0:
            print_model_structure(module, name, indent + 1)

def compare_loss():
    """Compare MLMLoss implementation."""
    print("="*80)
    print("Comparing MLMLoss Implementation")
    print("="*80)
    
    try:
        from losses.ssl_losses.mlm import MLMLoss as OurMLMLoss
        print("✓ Our MLMLoss imported successfully")
    except Exception as e:
        print(f"✗ Failed to import our MLMLoss: {e}")
        return
    
    try:
        from nemo.collections.asr.losses.ssl_losses.mlm import MLMLoss as NeMoMLMLoss
        print("✓ NeMo MLMLoss imported successfully")
    except Exception as e:
        print(f"✗ Failed to import NeMo MLMLoss: {e}")
        return
    
    # Compare implementations
    our_loss = OurMLMLoss(combine_time_steps=4, mask_threshold=0.8)
    nemo_loss = NeMoMLMLoss(combine_time_steps=4, mask_threshold=0.8)
    
    print("\nComparing forward method signatures...")
    import inspect
    our_sig = inspect.signature(our_loss.forward)
    nemo_sig = inspect.signature(nemo_loss.forward)
    
    if our_sig == nemo_sig:
        print("✓ Forward method signatures match")
    else:
        print("✗ Forward method signatures differ:")
        print(f"  Our: {our_sig}")
        print(f"  NeMo: {nemo_sig}")
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    batch_size = 2
    seq_len = 100
    feat_dim = 80
    
    masks = torch.rand(batch_size, feat_dim, seq_len) > 0.5
    masks = masks.float()
    decoder_outputs = torch.randn(batch_size, seq_len, 8192)
    decoder_outputs = torch.log_softmax(decoder_outputs, dim=-1)
    targets = torch.randint(0, 8192, (batch_size, seq_len))
    
    try:
        our_result = our_loss(masks=masks, decoder_outputs=decoder_outputs, targets=targets)
        print(f"✓ Our loss computed: {our_result.item():.4f}")
    except Exception as e:
        print(f"✗ Our loss failed: {e}")
    
    try:
        nemo_result = nemo_loss(masks=masks, decoder_outputs=decoder_outputs, targets=targets)
        print(f"✓ NeMo loss computed: {nemo_result.item():.4f}")
    except Exception as e:
        print(f"✗ NeMo loss failed: {e}")

def compare_conformer_encoder():
    """Compare ConformerEncoder structure and parameters."""
    print("\n" + "="*80)
    print("Comparing ConformerEncoder Structure")
    print("="*80)
    
    # Configuration matching NeMo NEST Fast-Conformer Large
    config = {
        'feat_in': 80,
        'n_layers': 17,
        'd_model': 512,
        'n_heads': 8,
        'conv_kernel_size': 9,
        'subsampling': 'dw_striding',
        'subsampling_factor': 8,
        'subsampling_conv_channels': 256,
        'ff_expansion_factor': 4,
        'self_attention_model': 'rel_pos',
        'xscaling': True,
        'untie_biases': True,
        'use_bias': True,
        'dropout': 0.1,
        'dropout_pre_encoder': 0.1,
        'dropout_emb': 0.0,
        'dropout_att': 0.1,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Our implementation
    try:
        from modules.conformer_encoder import ConformerEncoder as OurConformerEncoder
        print("\n✓ Our ConformerEncoder imported successfully")
        
        our_encoder = OurConformerEncoder(**config)
        our_params = count_parameters(our_encoder)
        print(f"✓ Our ConformerEncoder parameters: {our_params:,} ({our_params/1e6:.2f}M)")
        
        print("\nOur ConformerEncoder structure:")
        print_model_structure(our_encoder, "ConformerEncoder")
    except Exception as e:
        print(f"✗ Failed to create our ConformerEncoder: {e}")
        import traceback
        traceback.print_exc()
        our_params = None
    
    # NeMo implementation
    try:
        from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder as NeMoConformerEncoder
        print("\n✓ NeMo ConformerEncoder imported successfully")
        
        nemo_encoder = NeMoConformerEncoder(**config)
        nemo_params = count_parameters(nemo_encoder)
        print(f"✓ NeMo ConformerEncoder parameters: {nemo_params:,} ({nemo_params/1e6:.2f}M)")
        
        print("\nNeMo ConformerEncoder structure:")
        print_model_structure(nemo_encoder, "ConformerEncoder")
    except Exception as e:
        print(f"✗ Failed to create NeMo ConformerEncoder: {e}")
        import traceback
        traceback.print_exc()
        nemo_params = None
    
    # Compare
    if our_params is not None and nemo_params is not None:
        print("\n" + "="*80)
        print("Parameter Count Comparison")
        print("="*80)
        print(f"Our implementation:    {our_params:,} ({our_params/1e6:.2f}M)")
        print(f"NeMo implementation:   {nemo_params:,} ({nemo_params/1e6:.2f}M)")
        diff = abs(our_params - nemo_params)
        diff_pct = (diff / nemo_params) * 100
        print(f"Difference:            {diff:,} ({diff/1e6:.2f}M, {diff_pct:.2f}%)")
        
        if diff_pct < 1.0:
            print("✓ Parameter counts match closely!")
        else:
            print("⚠ Parameter counts differ significantly")
            
            # Try to find where the difference is
            print("\nComparing layer-by-layer...")
            our_layers = dict(our_encoder.named_modules())
            nemo_layers = dict(nemo_encoder.named_modules())
            
            for name in set(list(our_layers.keys()) + list(nemo_layers.keys())):
                if name == '':
                    continue
                our_layer = our_layers.get(name)
                nemo_layer = nemo_layers.get(name)
                
                if our_layer is None or nemo_layer is None:
                    continue
                
                our_p = count_parameters(our_layer) if our_layer is not None else 0
                nemo_p = count_parameters(nemo_layer) if nemo_layer is not None else 0
                
                if our_p != nemo_p and our_p > 0 and nemo_p > 0:
                    print(f"  {name}: Our={our_p:,}, NeMo={nemo_p:,}, Diff={abs(our_p-nemo_p):,}")

def compare_full_model():
    """Compare full SSL model structure."""
    print("\n" + "="*80)
    print("Comparing Full SSL Model")
    print("="*80)
    
    try:
        from omegaconf import OmegaConf
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'nest_fast-conformer.yaml')
        cfg = OmegaConf.load(config_path)
        
        print("✓ Configuration loaded")
        
        # Create our model
        from models.ssl_models import EncDecDenoiseMaskedTokenPredModel
        from lightning.pytorch import Trainer
        
        trainer = Trainer(devices=1, accelerator='cpu')
        our_model = EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=trainer)
        our_params = count_parameters(our_model)
        
        print(f"\n✓ Our full model created")
        print(f"  Total parameters: {our_params:,} ({our_params/1e6:.2f}M)")
        
        print("\nModel components:")
        print(f"  Encoder: {count_parameters(our_model.encoder):,} params")
        print(f"  Decoder: {count_parameters(our_model.decoder):,} params")
        print(f"  Quantizer: {count_parameters(our_model.quantizer):,} params")
        print(f"  Mask processor: {count_parameters(our_model.mask_processor):,} params")
        print(f"  Preprocessor: {count_parameters(our_model.preprocessor):,} params")
        
    except Exception as e:
        print(f"✗ Failed to create full model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Comparing our implementation with NeMo")
    print("="*80)
    
    # Compare loss
    compare_loss()
    
    # Compare encoder
    compare_conformer_encoder()
    
    # Compare full model
    compare_full_model()
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)

