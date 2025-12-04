#!/usr/bin/env python3
"""
Pure PyTorch training script - No Lightning, No OmegaConf dependency.

Usage:
    python train_pure_torch.py --config config/nest_fast-conformer.yaml --device cuda
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.ssl_model_pure_torch import PureTorchSSLModel


def create_optimizer_simple(model, cfg: dict, lr: float = None) -> torch.optim.Optimizer:
    """Create optimizer from plain dict config."""
    optim_cfg = cfg.get('optim', {})
    optim_name = str(optim_cfg.get('name', 'adamw')).lower()
    if lr is None:
        lr = float(optim_cfg.get('lr', 1e-4))
    else:
        lr = float(lr)
    betas = optim_cfg.get('betas', [0.9, 0.999])
    weight_decay = float(optim_cfg.get('weight_decay', 0.0))
    
    # Ensure betas are floats
    if isinstance(betas, (list, tuple)):
        betas = tuple(float(b) for b in betas)
    
    if optim_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optim_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optim_name == 'sgd':
        momentum = float(optim_cfg.get('momentum', 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")


def create_scheduler_simple(optimizer, cfg: dict):
    """Create scheduler from plain dict config."""
    optim_cfg = cfg.get('optim', {})
    sched_cfg = optim_cfg.get('sched', None)
    
    if sched_cfg is None:
        return None
    
    sched_name = str(sched_cfg.get('name', 'noam')).lower()
    
    if sched_name in ('noamannealing', 'noam'):
        d_model = int(sched_cfg.get('d_model', cfg.get('encoder', {}).get('d_model', 512)))
        warmup_steps = int(sched_cfg.get('warmup_steps', 10000))
        min_lr = float(sched_cfg.get('min_lr', 1e-6))
        
        def noam_lambda(step):
            step = max(step, 1)
            scale = d_model ** (-0.5)
            lr_scale = min(step ** (-0.5), step * warmup_steps ** (-1.5))
            return max(scale * lr_scale, min_lr / optimizer.defaults['lr'])
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, noam_lambda)
    
    elif sched_name == 'cosine':
        warmup_steps = int(sched_cfg.get('warmup_steps', 1000))
        max_steps = int(cfg.get('trainer', {}).get('max_steps', 100000))
        
        def cosine_with_warmup(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)
    
    return None
from data import ssl_dataset
from utils.logging import get_logger

logger = get_logger(__name__)


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config file and resolve interpolations."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Resolve simple interpolations like ${model.sample_rate}
    def resolve_interpolations(obj, root):
        if isinstance(obj, dict):
            return {k: resolve_interpolations(v, root) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_interpolations(v, root) for v in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            # Resolve interpolation
            path = obj[2:-1].split('.')
            val = root
            for p in path:
                val = val.get(p, obj)
                if val == obj:
                    break
            return val
        return obj
    
    return resolve_interpolations(cfg, cfg)


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
    
    # Load config using standard YAML (no OmegaConf)
    logger.info(f"Loading config from {args.config}")
    cfg = load_yaml_config(args.config)
    model_cfg = cfg.get('model', cfg)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model using from_config_file for proper parameter extraction
    logger.info("Creating model...")
    model = PureTorchSSLModel.from_config_file(args.config)
    model.to(device)
    
    # Create optimizer
    lr = args.lr if args.lr is not None else model_cfg.get('optim', {}).get('lr', 1e-4)
    optimizer = create_optimizer_simple(model, model_cfg, lr=lr)
    
    # Create scheduler
    scheduler = create_scheduler_simple(optimizer, model_cfg)
    
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
    
    if 'train_ds' in model_cfg and model_cfg['train_ds'] is not None:
        train_loader = create_dataloader(model_cfg['train_ds'], shuffle=True)
        if train_loader:
            logger.info(f"Train dataloader: {len(train_loader)} batches")
    
    if 'validation_ds' in model_cfg and model_cfg['validation_ds'] is not None:
        val_loader = create_dataloader(model_cfg['validation_ds'], shuffle=False)
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

