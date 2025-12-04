#!/usr/bin/env python3
"""
Example script showing how to use SimplifiedSSLModel.

This demonstrates:
1. Creating the simplified model
2. Training with the standard training script
3. Verifying alignment with original model
"""

import torch
from omegaconf import OmegaConf
from pathlib import Path
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def example_create_model():
    """Example: Create SimplifiedSSLModel from config."""
    from models.ssl_models_simplified import SimplifiedSSLModel
    
    # Load config
    config_path = os.path.join(project_root, "config", "nest_fast-conformer.yaml")
    cfg = OmegaConf.load(config_path)
    
    # Create model (no trainer needed for inference)
    model = SimplifiedSSLModel(cfg=cfg.model, trainer=None)
    
    print(f"Model created: {type(model).__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def example_forward_pass():
    """Example: Run forward pass with dummy data."""
    from models.ssl_models_simplified import SimplifiedSSLModel
    from data.ssl_dataset import AudioNoiseBatch
    
    # Load config
    config_path = os.path.join(project_root, "config", "nest_fast-conformer.yaml")
    cfg = OmegaConf.load(config_path)
    
    # Create model
    model = SimplifiedSSLModel(cfg=cfg.model, trainer=None)
    model.eval()
    
    # Create dummy batch
    batch_size = 2
    audio_len = 16000
    
    batch = AudioNoiseBatch(
        audio=torch.randn(batch_size, audio_len),
        audio_len=torch.tensor([audio_len, audio_len], dtype=torch.int32),
        noise=torch.randn(batch_size, audio_len),
        noise_len=torch.tensor([audio_len, audio_len], dtype=torch.int32),
        noisy_audio=torch.randn(batch_size, audio_len),
        noisy_audio_len=torch.tensor([audio_len, audio_len], dtype=torch.int32),
    )
    
    # Forward pass
    with torch.no_grad():
        log_probs, encoded_len, masks, tokens = model.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=True,
        )
    
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Encoded len: {encoded_len}")
    print(f"Masks shape: {masks.shape}")
    print(f"Tokens shape: {tokens.shape}")


def example_training():
    """
    Example: Training with SimplifiedSSLModel.
    
    Run this command from the nest_ssl_project directory:
    
    ```bash
    python train_simplified.py \
        model.train_ds.manifest_filepath=<path to train manifest> \
        model.train_ds.noise_manifest=<path to noise manifest> \
        model.validation_ds.manifest_filepath=<path to val manifest> \
        model.validation_ds.noise_manifest=<path to noise manifest> \
        trainer.devices=1 \
        trainer.accelerator="gpu" \
        trainer.max_epochs=10
    ```
    """
    print(__doc__)
    print(example_training.__doc__)


def example_verify_alignment():
    """
    Example: Verify that SimplifiedSSLModel aligns with original.
    
    Run this command:
    
    ```bash
    python tools/verify_simplified_alignment.py --device cpu
    ```
    
    This will:
    1. Create both original and simplified models
    2. Copy weights from original to simplified
    3. Run forward pass with same input
    4. Compare outputs bit-for-bit
    """
    print(example_verify_alignment.__doc__)


if __name__ == '__main__':
    print("""
SimplifiedSSLModel Examples
===========================

The SimplifiedSSLModel is a streamlined version of EncDecDenoiseMaskedTokenPredModel that:
- Maintains bit-exact forward pass alignment with the original
- Supports optimizer/scheduler configuration via ModelPT
- Simplifies data loader setup
- Removes unnecessary mixins while keeping core functionality

Available examples:

1. Create model:
   from examples.use_simplified_ssl_model import example_create_model
   model = example_create_model()

2. Run forward pass:
   from examples.use_simplified_ssl_model import example_forward_pass
   example_forward_pass()

3. Training:
   python train_simplified.py --help

4. Verify alignment:
   python tools/verify_simplified_alignment.py --device cpu

Files:
- models/ssl_models_simplified.py - SimplifiedSSLModel class
- train_simplified.py - Training script
- tools/verify_simplified_alignment.py - Alignment verification
    """)
