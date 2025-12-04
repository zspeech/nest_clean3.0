#!/usr/bin/env python3
"""
Example script showing how to use SimplifiedSSLModel.

This demonstrates:
1. How to create the simplified model
2. How to set up data loaders (users handle their own)
3. How to train with PyTorch Lightning
"""

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path

# Import the simplified model
from models.ssl_models_simplified import SimplifiedSSLModel

# Import data utilities (users can use their own data loading)
from data import ssl_dataset


def create_simplified_model(cfg_path: str = None, cfg: DictConfig = None):
    """
    Create a SimplifiedSSLModel instance.
    
    Args:
        cfg_path: Path to config file (optional)
        cfg: Config dict (optional, if cfg_path not provided)
    
    Returns:
        SimplifiedSSLModel instance
    """
    if cfg_path:
        cfg = OmegaConf.load(cfg_path)
    
    if cfg is None:
        raise ValueError("Either cfg_path or cfg must be provided")
    
    # Create model
    model = SimplifiedSSLModel(cfg=cfg.model)
    
    return model


def example_training_with_custom_dataloader():
    """
    Example: Training with custom data loader.
    Users handle their own data loading.
    """
    # Load config
    cfg = OmegaConf.load("config/nest_fast-conformer.yaml")  # Adjust path as needed
    
    # Create model
    model = SimplifiedSSLModel(cfg=cfg.model)
    
    # Users create their own data loaders
    # Example: Using the existing SSL dataset utilities
    train_dataset = ssl_dataset.get_audio_noise_dataset_from_config(
        cfg.model.train_ds,
        global_rank=0,  # Adjust for DDP
        world_size=1,   # Adjust for DDP
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.model.train_ds.batch_size,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None,
        shuffle=True,
        num_workers=4,
    )
    
    val_dataset = ssl_dataset.get_audio_noise_dataset_from_config(
        cfg.model.validation_ds,
        global_rank=0,
        world_size=1,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.model.validation_ds.batch_size,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None,
        shuffle=False,
        num_workers=4,
    )
    
    # Configure optimizer (users can override configure_optimizers or set it up manually)
    # For now, we'll use a simple setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.optim.lr,
        betas=cfg.model.optim.betas,
        weight_decay=cfg.model.optim.weight_decay,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        # ... other trainer configs
    )
    
    # Train
    trainer.fit(model, train_dataloader, val_dataloader)


def example_forward_only():
    """
    Example: Using the model for forward pass only (inference).
    """
    # Load config
    cfg = OmegaConf.load("config/nest_fast-conformer.yaml")
    
    # Create model
    model = SimplifiedSSLModel(cfg=cfg.model)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    audio_length = 16000
    audio = torch.randn(batch_size, audio_length)
    audio_len = torch.tensor([audio_length, audio_length])
    noisy_audio = torch.randn(batch_size, audio_length)
    noisy_audio_len = torch.tensor([audio_length, audio_length])
    
    # Forward pass
    with torch.no_grad():
        log_probs, encoded_len, masks, tokens = model.forward(
            input_signal=audio,
            input_signal_length=audio_len,
            noisy_input_signal=noisy_audio,
            noisy_input_signal_length=noisy_audio_len,
            apply_mask=True,
        )
    
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Encoded len: {encoded_len}")
    print(f"Masks shape: {masks.shape}")
    print(f"Tokens shape: {tokens.shape}")


def example_with_custom_batch_format():
    """
    Example: Using custom batch format (tuple/list instead of AudioNoiseBatch).
    """
    cfg = OmegaConf.load("config/nest_fast-conformer.yaml")
    model = SimplifiedSSLModel(cfg=cfg.model)
    
    # Custom batch format: (audio, audio_len, noise, noise_len, noisy_audio, noisy_audio_len)
    batch_size = 2
    audio_length = 16000
    
    batch = (
        torch.randn(batch_size, audio_length),  # audio
        torch.tensor([audio_length, audio_length]),  # audio_len
        torch.randn(batch_size, audio_length),  # noise
        torch.tensor([audio_length, audio_length]),  # noise_len
        torch.randn(batch_size, audio_length),  # noisy_audio
        torch.tensor([audio_length, audio_length]),  # noisy_audio_len
    )
    
    # Training step accepts this format
    output = model.training_step(batch, batch_idx=0)
    print(f"Loss: {output['loss']}")


if __name__ == '__main__':
    print("""
    SimplifiedSSLModel Usage Examples:
    
    1. Forward pass only:
       python -c "from examples.use_simplified_ssl_model import example_forward_only; example_forward_only()"
    
    2. Training with custom dataloader:
       python -c "from examples.use_simplified_ssl_model import example_training_with_custom_dataloader; example_training_with_custom_dataloader()"
    
    3. Custom batch format:
       python -c "from examples.use_simplified_ssl_model import example_with_custom_batch_format; example_with_custom_batch_format()"
    """)

