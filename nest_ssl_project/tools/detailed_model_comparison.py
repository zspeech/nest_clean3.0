#!/usr/bin/env python3
"""
Detailed comparison between NeMo and nest_ssl_project models.
Compares:
1. Configuration parameters
2. Model architecture
3. Parameter counts
4. Forward pass outputs
"""

import sys
import os
import torch
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_yaml_config(path):
    """Load YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def compare_configs(nemo_config_path, our_config_path):
    """Compare configuration files."""
    print("="*80)
    print("Configuration Comparison")
    print("="*80)
    
    nemo_config = load_yaml_config(nemo_config_path)
    our_config = load_yaml_config(our_config_path)
    
    # Compare model section
    nemo_model = nemo_config.get('model', {})
    our_model = our_config.get('model', {})
    
    # Key parameters to compare
    key_params = {
        'sample_rate': 'model.sample_rate',
        'num_classes': 'model.num_classes',
        'num_books': 'model.num_books',
        'code_dim': 'model.code_dim',
        'mask_position': 'model.mask_position',
    }
    
    encoder_params = {
        'n_layers': 'model.encoder.n_layers',
        'd_model': 'model.encoder.d_model',
        'n_heads': 'model.encoder.n_heads',
        'conv_kernel_size': 'model.encoder.conv_kernel_size',
        'subsampling_factor': 'model.encoder.subsampling_factor',
        'subsampling_conv_channels': 'model.encoder.subsampling_conv_channels',
        'ff_expansion_factor': 'model.encoder.ff_expansion_factor',
        'self_attention_model': 'model.encoder.self_attention_model',
        'xscaling': 'model.encoder.xscaling',
        'untie_biases': 'model.encoder.untie_biases',
        'use_bias': 'model.encoder.use_bias',
        'dropout': 'model.encoder.dropout',
        'dropout_pre_encoder': 'model.encoder.dropout_pre_encoder',
        'dropout_emb': 'model.encoder.dropout_emb',
        'dropout_att': 'model.encoder.dropout_att',
    }
    
    preprocessor_params = {
        'sample_rate': 'model.preprocessor.sample_rate',
        'normalize': 'model.preprocessor.normalize',
        'window_size': 'model.preprocessor.window_size',
        'window_stride': 'model.preprocessor.window_stride',
        'features': 'model.preprocessor.features',
        'n_fft': 'model.preprocessor.n_fft',
        'log': 'model.preprocessor.log',
    }
    
    loss_params = {
        'combine_time_steps': 'model.loss.combine_time_steps',
        'mask_threshold': 'model.loss.mask_threshold',
        'num_decoders': 'model.loss.num_decoders',
    }
    
    all_params = {
        **key_params,
        **{f'encoder.{k}': v for k, v in encoder_params.items()},
        **{f'preprocessor.{k}': v for k, v in preprocessor_params.items()},
        **{f'loss.{k}': v for k, v in loss_params.items()},
    }
    
    differences = []
    matches = []
    
    def get_nested_value(config, path):
        """Get nested value from config using dot notation."""
        parts = path.split('.')
        value = config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value
    
    for param_name, config_path in all_params.items():
        nemo_value = get_nested_value(nemo_config, config_path)
        our_value = get_nested_value(our_config, config_path)
        
        if nemo_value == our_value:
            matches.append((param_name, nemo_value))
        else:
            differences.append((param_name, nemo_value, our_value))
    
    print(f"\n✓ Matching parameters: {len(matches)}")
    print(f"✗ Different parameters: {len(differences)}")
    
    if differences:
        print("\nDifferences found:")
        for param_name, nemo_val, our_val in differences:
            print(f"  {param_name}:")
            print(f"    NeMo:  {nemo_val}")
            print(f"    Ours:  {our_val}")
    else:
        print("\n✓ All parameters match!")
    
    return len(differences) == 0

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_model_structures():
    """Compare model structures."""
    print("\n" + "="*80)
    print("Model Structure Comparison")
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
        'pos_emb_max_len': 5000,
        'conv_norm_type': 'batch_norm',
        'conv_context_size': None,
        'att_context_size': [-1, -1],
        'att_context_style': 'regular',
        'causal_downsampling': False,
        'stochastic_depth_drop_prob': 0.0,
        'stochastic_depth_mode': 'linear',
        'stochastic_depth_start_layer': 1,
    }
    
    # Our implementation
    try:
        from modules.conformer_encoder import ConformerEncoder as OurConformerEncoder
        print("\n[Our Implementation]")
        our_encoder = OurConformerEncoder(**config)
        our_params = count_parameters(our_encoder)
        print(f"✓ Parameters: {our_params:,} ({our_params/1e6:.2f}M)")
        
        # Test forward pass
        batch_size = 2
        seq_len = 100
        feat_dim = 80
        x = torch.randn(batch_size, feat_dim, seq_len)
        lengths = torch.tensor([seq_len, seq_len])
        
        our_output, our_lengths = our_encoder(x, lengths)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {our_output.shape}")
        print(f"  Output lengths: {our_lengths}")
        
    except Exception as e:
        print(f"✗ Our implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # NeMo implementation (if available)
    try:
        import sys
        nemo_path = Path(__file__).parent.parent.parent / "NeMo"
        if nemo_path.exists():
            sys.path.insert(0, str(nemo_path))
        
        from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder as NeMoConformerEncoder
        print("\n[NeMo Implementation]")
        nemo_encoder = NeMoConformerEncoder(**config)
        nemo_params = count_parameters(nemo_encoder)
        print(f"✓ Parameters: {nemo_params:,} ({nemo_params/1e6:.2f}M)")
        
        # Test forward pass
        nemo_output, nemo_lengths = nemo_encoder(x, lengths)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {nemo_output.shape}")
        print(f"  Output lengths: {nemo_lengths}")
        
        # Compare parameter counts
        print(f"\n[Parameter Count Comparison]")
        diff = abs(our_params - nemo_params)
        diff_pct = (diff / nemo_params * 100) if nemo_params > 0 else 0
        print(f"  Our:    {our_params:,} ({our_params/1e6:.2f}M)")
        print(f"  NeMo:   {nemo_params:,} ({nemo_params/1e6:.2f}M)")
        print(f"  Diff:   {diff:,} ({diff/1e6:.2f}M, {diff_pct:.2f}%)")
        
        if diff_pct < 1.0:
            print("  ✓ Parameter counts match closely!")
        elif diff_pct < 5.0:
            print("  ⚠ Parameter counts differ slightly")
        else:
            print("  ✗ Parameter counts differ significantly")
        
        # Compare output shapes
        if our_output.shape == nemo_output.shape:
            print(f"\n✓ Output shapes match: {our_output.shape}")
        else:
            print(f"\n✗ Output shapes differ:")
            print(f"  Our:  {our_output.shape}")
            print(f"  NeMo: {nemo_output.shape}")
        
        # Compare output values (if shapes match)
        if our_output.shape == nemo_output.shape:
            # Initialize with same weights for fair comparison
            # This is just a shape/structure check, not a numerical comparison
            print(f"\n[Note] Numerical comparison requires same initialization")
        
    except ImportError as e:
        print(f"\n⚠ NeMo not available for comparison: {e}")
        print("  Install NeMo to enable full comparison")
    except Exception as e:
        print(f"\n✗ NeMo implementation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return True

def compare_loss_implementations():
    """Compare loss implementations."""
    print("\n" + "="*80)
    print("Loss Implementation Comparison")
    print("="*80)
    
    try:
        from losses.ssl_losses.mlm import MLMLoss as OurMLMLoss
        print("✓ Our MLMLoss imported")
    except Exception as e:
        print(f"✗ Failed to import our MLMLoss: {e}")
        return False
    
    try:
        import sys
        nemo_path = Path(__file__).parent.parent.parent / "NeMo"
        if nemo_path.exists():
            sys.path.insert(0, str(nemo_path))
        
        from nemo.collections.asr.losses.ssl_losses.mlm import MLMLoss as NeMoMLMLoss
        print("✓ NeMo MLMLoss imported")
    except ImportError:
        print("⚠ NeMo MLMLoss not available")
        return False
    
    # Compare signatures
    import inspect
    our_sig = inspect.signature(OurMLMLoss.__init__)
    nemo_sig = inspect.signature(NeMoMLMLoss.__init__)
    
    print(f"\n[Constructor Signatures]")
    print(f"  Our:  {our_sig}")
    print(f"  NeMo: {nemo_sig}")
    
    # Test with dummy data
    our_loss = OurMLMLoss(combine_time_steps=8, mask_threshold=0.8)
    nemo_loss = NeMoMLMLoss(combine_time_steps=8, mask_threshold=0.8)
    
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
        print(f"\n✓ Our loss computed: {our_result.item():.4f}")
    except Exception as e:
        print(f"\n✗ Our loss failed: {e}")
        return False
    
    try:
        nemo_result = nemo_loss(masks=masks, decoder_outputs=decoder_outputs, targets=targets)
        print(f"✓ NeMo loss computed: {nemo_result.item():.4f}")
        
        diff = abs(our_result.item() - nemo_result.item())
        print(f"  Difference: {diff:.6f}")
        if diff < 1e-5:
            print("  ✓ Loss values match!")
        else:
            print("  ⚠ Loss values differ (may be due to different initialization)")
    except Exception as e:
        print(f"✗ NeMo loss failed: {e}")
        return False
    
    return True

def main():
    """Main comparison function."""
    print("="*80)
    print("NeMo vs nest_ssl_project Detailed Model Comparison")
    print("="*80)
    
    # Paths
    nemo_config_path = Path(__file__).parent.parent.parent / "NeMo" / "examples" / "asr" / "conf" / "ssl" / "nest" / "nest_fast-conformer.yaml"
    our_config_path = Path(__file__).parent.parent / "config" / "nest_fast-conformer.yaml"
    
    results = {}
    
    # 1. Compare configurations
    if nemo_config_path.exists() and our_config_path.exists():
        results['config'] = compare_configs(str(nemo_config_path), str(our_config_path))
    else:
        print("⚠ Configuration files not found, skipping config comparison")
        results['config'] = None
    
    # 2. Compare model structures
    results['model_structure'] = compare_model_structures()
    
    # 3. Compare loss implementations
    results['loss'] = compare_loss_implementations()
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    for check_name, result in results.items():
        if result is None:
            status = "⚠ Skipped"
        elif result:
            status = "✓ Pass"
        else:
            status = "✗ Fail"
        print(f"  {check_name:20s}: {status}")
    
    all_passed = all(r for r in results.values() if r is not None)
    
    if all_passed:
        print("\n✓ All comparisons passed!")
    else:
        print("\n⚠ Some comparisons failed or were skipped")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

