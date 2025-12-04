#!/usr/bin/env python3
"""
Pure PyTorch training script - No Lightning dependency.

Usage:
    python train_pure_torch.py --config config/nest_fast-conformer.yaml --device cuda
"""

import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.ssl_model_pure_torch import PureTorchSSLModel, create_optimizer, create_scheduler
from data import ssl_dataset
from utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(cfg, shuffle=True):
    """Create dataloader from config."""
    dataset = ssl_dataset.get_audio_noise_dataset_from_config(
        cfg,
        global_rank=0,
        world_size=1,
    )
    
    if dataset is None:
        return None
    
    collate_fn = None
    if hasattr(dataset, 'collate_fn'):
        collate_fn = dataset.collate_fn
    
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.get('batch_size', 4),
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=cfg.get('pin_memory', False),
        drop_last=cfg.get('drop_last', False),
    )


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, log_interval=10):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        if hasattr(batch, 'audio'):
            batch = ssl_dataset.AudioNoiseBatch(
                audio=batch.audio.to(device),
                audio_len=batch.audio_len.to(device),
                noise=batch.noise.to(device) if batch.noise is not None else None,
                noise_len=batch.noise_len.to(device) if batch.noise_len is not None else None,
                noisy_audio=batch.noisy_audio.to(device),
                noisy_audio_len=batch.noisy_audio_len.to(device),
            )
        
        # Forward + backward
        optimizer.zero_grad()
        loss = model.training_step(batch)
        loss.backward()
        
        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })
    
    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def validate(model, dataloader, device):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        # Move batch to device
        if hasattr(batch, 'audio'):
            batch = ssl_dataset.AudioNoiseBatch(
                audio=batch.audio.to(device),
                audio_len=batch.audio_len.to(device),
                noise=batch.noise.to(device) if batch.noise is not None else None,
                noise_len=batch.noise_len.to(device) if batch.noise_len is not None else None,
                noisy_audio=batch.noisy_audio.to(device),
                noisy_audio_len=batch.noisy_audio_len.to(device),
            )
        
        loss = model.validation_step(batch)
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Pure PyTorch SSL Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    device = torch.device(device)
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    cfg = OmegaConf.load(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    logger.info("Creating model...")
    model = PureTorchSSLModel(cfg.model)
    model.to(device)
    
    # Create optimizer
    optim_cfg = cfg.model.get('optim', {'name': 'adamw', 'lr': 1e-4})
    if args.lr is not None:
        optim_cfg.lr = args.lr
    optimizer = create_optimizer(model, optim_cfg)
    
    # Create scheduler
    sched_cfg = optim_cfg.get('sched', None)
    scheduler = create_scheduler(optimizer, sched_cfg) if sched_cfg else None
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        info = model.load_checkpoint(args.resume, optimizer)
        start_epoch = info.get('epoch', 0)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = None
    val_loader = None
    
    if 'train_ds' in cfg.model and cfg.model.train_ds is not None:
        train_loader = create_dataloader(cfg.model.train_ds, shuffle=True)
        if train_loader:
            logger.info(f"Train dataloader: {len(train_loader)} batches")
    
    if 'validation_ds' in cfg.model and cfg.model.validation_ds is not None:
        val_loader = create_dataloader(cfg.model.validation_ds, shuffle=False)
        if val_loader:
            logger.info(f"Validation dataloader: {len(val_loader)} batches")
    
    # Training settings
    num_epochs = args.epochs if args.epochs else cfg.get('trainer', {}).get('max_epochs', 10)
    
    logger.info(f"Starting training for {num_epochs} epochs on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        if train_loader:
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            logger.info(f"Epoch {epoch}: train_loss = {train_loss:.4f}")
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, device)
            logger.info(f"Epoch {epoch}: val_loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_checkpoint(
                    output_dir / "best_model.pt",
                    optimizer=optimizer,
                    epoch=epoch,
                )
        
        # Save checkpoint every epoch
        model.save_checkpoint(
            output_dir / f"checkpoint_epoch_{epoch}.pt",
            optimizer=optimizer,
            epoch=epoch,
        )
    
    # Save final model
    model.save_checkpoint(
        output_dir / "final_model.pt",
        optimizer=optimizer,
        epoch=num_epochs,
    )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

