#!/usr/bin/env python3
"""
Checkpoint Compatibility Verification Script

This script verifies that checkpoints saved by the local implementation
can be loaded by NeMo, and vice versa.

Usage:
    python tools/verify_checkpoint_compatibility.py --ckpt_path <path_to_checkpoint>
    
    # Test loading NeMo checkpoint in local model
    python tools/verify_checkpoint_compatibility.py --ckpt_path nemo_checkpoint.nemo --direction nemo_to_local
    
    # Test loading local checkpoint in NeMo model
    python tools/verify_checkpoint_compatibility.py --ckpt_path local_checkpoint.nemo --direction local_to_nemo
    
    # Test bidirectional (default)
    python tools/verify_checkpoint_compatibility.py --ckpt_path checkpoint.nemo --direction bidirectional
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

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("Warning: NeMo not available. Only local model loading will be tested.")

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


def load_nemo_model_from_checkpoint(ckpt_path: str):
    """Load NeMo model from checkpoint."""
    if not NEMO_AVAILABLE:
        print("✗ NeMo not available, skipping NeMo model loading test")
        return None, False
    
    print(f"\n{'='*60}")
    print(f"Loading NEMO model from checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    
    try:
        # Try loading as .nemo file
        if ckpt_path.endswith('.nemo'):
            # Try EncDecDenoiseMaskedTokenPredModel first
            try:
                model = nemo_asr.models.EncDecDenoiseMaskedTokenPredModel.restore_from(ckpt_path)
                print(f"✓ Successfully loaded as EncDecDenoiseMaskedTokenPredModel")
            except:
                # Try base SSL model
                model = nemo_asr.models.SpeechEncDecSelfSupervisedModel.restore_from(ckpt_path)
                print(f"✓ Successfully loaded as SpeechEncDecSelfSupervisedModel")
        else:
            # Try loading as PyTorch Lightning checkpoint
            model = nemo_asr.models.EncDecDenoiseMaskedTokenPredModel.load_from_checkpoint(ckpt_path)
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


def test_nemo_to_local(ckpt_path: str):
    """Test loading NeMo checkpoint in local model."""
    print(f"\n{'#'*60}")
    print(f"TEST 1: Loading NeMo checkpoint in LOCAL model")
    print(f"{'#'*60}")
    
    local_model, success = load_local_model_from_checkpoint(ckpt_path)
    
    if success:
        print(f"\n✓ SUCCESS: NeMo checkpoint can be loaded in local model")
        return True
    else:
        print(f"\n✗ FAILED: NeMo checkpoint cannot be loaded in local model")
        return False


def test_local_to_nemo(ckpt_path: str):
    """Test loading local checkpoint in NeMo model."""
    if not NEMO_AVAILABLE:
        print(f"\n{'#'*60}")
        print(f"TEST 2: Loading LOCAL checkpoint in NEMO model")
        print(f"{'#'*60}")
        print("✗ SKIPPED: NeMo not available")
        return None
    
    print(f"\n{'#'*60}")
    print(f"TEST 2: Loading LOCAL checkpoint in NEMO model")
    print(f"{'#'*60}")
    
    nemo_model, success = load_nemo_model_from_checkpoint(ckpt_path)
    
    if success:
        print(f"\n✓ SUCCESS: Local checkpoint can be loaded in NeMo model")
        return True
    else:
        print(f"\n✗ FAILED: Local checkpoint cannot be loaded in NeMo model")
        return False


def test_bidirectional(ckpt_path: str):
    """Test bidirectional loading and compare state dicts."""
    print(f"\n{'#'*60}")
    print(f"BIDIRECTIONAL TEST: Loading checkpoint in both models and comparing")
    print(f"{'#'*60}")
    
    # Load in local model
    local_model, local_success = load_local_model_from_checkpoint(ckpt_path)
    
    # Load in NeMo model (if available)
    nemo_model = None
    nemo_success = False
    if NEMO_AVAILABLE:
        nemo_model, nemo_success = load_nemo_model_from_checkpoint(ckpt_path)
    
    # Compare state dicts if both loaded successfully
    if local_success and nemo_success and local_model is not None and nemo_model is not None:
        local_sd = local_model.state_dict()
        nemo_sd = nemo_model.state_dict()
        
        are_identical = compare_state_dicts(local_sd, nemo_sd, "Local Model", "NeMo Model")
        
        if are_identical:
            print(f"\n✓ SUCCESS: Checkpoints are fully compatible!")
            return True
        else:
            print(f"\n⚠ WARNING: Checkpoints can be loaded but state dicts differ")
            return False
    elif local_success:
        print(f"\n✓ SUCCESS: Local model can load checkpoint")
        return True
    elif nemo_success:
        print(f"\n✓ SUCCESS: NeMo model can load checkpoint")
        return True
    else:
        print(f"\n✗ FAILED: Neither model could load checkpoint")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify checkpoint compatibility between local implementation and NeMo"
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        required=True,
        help='Path to checkpoint file (.nemo or .ckpt)'
    )
    parser.add_argument(
        '--direction',
        type=str,
        default='bidirectional',
        choices=['nemo_to_local', 'local_to_nemo', 'bidirectional'],
        help='Test direction: nemo_to_local, local_to_nemo, or bidirectional (default)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model config file (optional, uses default if not provided)'
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
    print(f"Checkpoint Compatibility Verification")
    print(f"{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Direction: {args.direction}")
    print(f"NeMo available: {NEMO_AVAILABLE}")
    print(f"{'='*60}")
    
    # Run tests based on direction
    success = False
    
    if args.direction == 'nemo_to_local':
        success = test_nemo_to_local(ckpt_path)
    elif args.direction == 'local_to_nemo':
        success = test_local_to_nemo(ckpt_path)
    else:  # bidirectional
        success = test_bidirectional(ckpt_path)
    
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

