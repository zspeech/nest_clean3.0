#!/usr/bin/env python3
"""
Checkpoint Compatibility Verification Script

This script verifies that checkpoints saved by the local implementation
can be correctly loaded back.

Usage:
    python tools/verify_checkpoint_compatibility.py --ckpt_path <path_to_checkpoint>
    
    # Test loading and saving round-trip
    python tools/verify_checkpoint_compatibility.py --ckpt_path checkpoint.nemo --test_save_load
"""

import argparse
import sys
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'nest_ssl_project'))

from models.ssl_models import EncDecDenoiseMaskedTokenPredModel
from utils.logging import get_logger

logger = get_logger(__name__)


def get_model_config():
    """Get default model config for testing."""
    config_path = project_root / 'nest_ssl_project' / 'config' / 'nest_fast-conformer.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    return cfg.model


def load_local_model_from_checkpoint(ckpt_path: str, config=None):
    """Load local model from checkpoint."""
    print(f"\n{'='*60}")
    print(f"Loading LOCAL model from checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    
    if config is None:
        config = get_model_config()
    
    try:
        # Try loading as .nemo file
        if ckpt_path.endswith('.nemo'):
            model = EncDecDenoiseMaskedTokenPredModel.restore_from(ckpt_path)
            print(f"✓ Successfully loaded .nemo checkpoint")
        else:
            # Try loading as PyTorch Lightning checkpoint
            model = EncDecDenoiseMaskedTokenPredModel.load_from_checkpoint(
                ckpt_path,
                cfg=config,
                strict=False
            )
            print(f"✓ Successfully loaded Lightning checkpoint")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model, True
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def save_and_reload_model(model, save_path: str):
    """Save model and reload it to verify round-trip compatibility."""
    print(f"\n{'='*60}")
    print(f"Testing save and reload round-trip")
    print(f"{'='*60}")
    
    try:
        # Save original state dict
        original_sd = model.state_dict()
        
        # Save model
        model.save_to(save_path)
        print(f"✓ Model saved to {save_path}")
        
        # Reload model
        reloaded_model, success = load_local_model_from_checkpoint(save_path)
        
        if success and reloaded_model is not None:
            # Compare state dicts
            reloaded_sd = reloaded_model.state_dict()
            are_identical = compare_state_dicts(original_sd, reloaded_sd, "Original Model", "Reloaded Model")
            
            if are_identical:
                print(f"\n✓ SUCCESS: Save and reload round-trip works correctly!")
                return True
            else:
                print(f"\n✗ FAILED: State dicts differ after reload")
                return False
        else:
            print(f"\n✗ FAILED: Could not reload saved model")
            return False
    except Exception as e:
        print(f"✗ Failed to test save/reload: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_state_dicts(sd1: dict, sd2: dict, name1: str = "Model 1", name2: str = "Model 2"):
    """Compare two state dictionaries."""
    print(f"\n{'='*60}")
    print(f"Comparing state dicts: {name1} vs {name2}")
    print(f"{'='*60}")
    
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    print(f"\nKeys comparison:")
    print(f"  Common keys: {len(common_keys)}")
    print(f"  Only in {name1}: {len(only_in_1)}")
    print(f"  Only in {name2}: {len(only_in_2)}")
    
    if only_in_1:
        print(f"\n  Keys only in {name1}:")
        for key in sorted(list(only_in_1))[:10]:  # Show first 10
            print(f"    - {key}")
        if len(only_in_1) > 10:
            print(f"    ... and {len(only_in_1) - 10} more")
    
    if only_in_2:
        print(f"\n  Keys only in {name2}:")
        for key in sorted(list(only_in_2))[:10]:  # Show first 10
            print(f"    - {key}")
        if len(only_in_2) > 10:
            print(f"    ... and {len(only_in_2) - 10} more")
    
    # Compare common keys
    mismatched_shapes = []
    mismatched_values = []
    
    for key in sorted(common_keys):
        tensor1 = sd1[key]
        tensor2 = sd2[key]
        
        if tensor1.shape != tensor2.shape:
            mismatched_shapes.append((key, tensor1.shape, tensor2.shape))
        elif not torch.allclose(tensor1, tensor2, atol=1e-5):
            mismatched_values.append(key)
    
    print(f"\nShape mismatches: {len(mismatched_shapes)}")
    if mismatched_shapes:
        for key, shape1, shape2 in mismatched_shapes[:5]:  # Show first 5
            print(f"  - {key}: {shape1} vs {shape2}")
        if len(mismatched_shapes) > 5:
            print(f"    ... and {len(mismatched_shapes) - 5} more")
    
    print(f"\nValue mismatches: {len(mismatched_values)}")
    if mismatched_values:
        for key in mismatched_values[:5]:  # Show first 5
            print(f"  - {key}")
        if len(mismatched_values) > 5:
            print(f"    ... and {len(mismatched_values) - 5} more")
    
    # Summary
    if len(only_in_1) == 0 and len(only_in_2) == 0 and len(mismatched_shapes) == 0 and len(mismatched_values) == 0:
        print(f"\n✓ State dicts are identical!")
        return True
    else:
        print(f"\n✗ State dicts differ")
        return False


def test_load_checkpoint(ckpt_path: str, config=None):
    """Test loading checkpoint in local model."""
    print(f"\n{'#'*60}")
    print(f"TEST: Loading checkpoint in local model")
    print(f"{'#'*60}")
    
    local_model, success = load_local_model_from_checkpoint(ckpt_path, config)
    
    if success:
        print(f"\n✓ SUCCESS: Checkpoint can be loaded in local model")
        return local_model, True
    else:
        print(f"\n✗ FAILED: Checkpoint cannot be loaded in local model")
        return None, False


def main():
    parser = argparse.ArgumentParser(
        description="Verify checkpoint loading for local implementation"
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        required=True,
        help='Path to checkpoint file (.nemo or .ckpt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model config file (optional, uses default if not provided)'
    )
    parser.add_argument(
        '--test_save_load',
        action='store_true',
        help='Test save and reload round-trip (saves to <ckpt_path>.test.nemo)'
    )
    
    args = parser.parse_args()
    
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint file not found: {ckpt_path}")
        sys.exit(1)
    
    ckpt_path = str(ckpt_path.resolve())
    
    # Load config if provided
    config = None
    if args.config:
        config = OmegaConf.load(args.config)
    
    print(f"\n{'='*60}")
    print(f"Checkpoint Loading Verification")
    print(f"{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test save/reload: {args.test_save_load}")
    print(f"{'='*60}")
    
    # Test loading checkpoint
    model, success = test_load_checkpoint(ckpt_path, config)
    
    # Test save/reload if requested
    if success and args.test_save_load and model is not None:
        test_save_path = ckpt_path.replace('.nemo', '.test.nemo').replace('.ckpt', '.test.nemo')
        save_success = save_and_reload_model(model, test_save_path)
        success = success and save_success
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    if success:
        print(f"✓ Verification PASSED")
        sys.exit(0)
    else:
        print(f"✗ Verification FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()

