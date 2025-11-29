# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simplified experiment manager to replace nemo.utils.exp_manager
"""

import os
from pathlib import Path
from typing import Optional
import lightning.pytorch as pl
from omegaconf import DictConfig

from utils.logging import get_logger

logger = get_logger(__name__)


def exp_manager(trainer: pl.Trainer, exp_cfg: Optional[DictConfig] = None):
    """
    Setup experiment manager for PyTorch Lightning.
    Simplified version that sets up logging and checkpointing.
    
    Args:
        trainer: PyTorch Lightning Trainer instance
        exp_cfg: Experiment configuration (optional)
    """
    if exp_cfg is None:
        return
    
    # Setup experiment directory
    exp_dir = exp_cfg.get('exp_dir', None)
    if exp_dir is None:
        exp_dir = Path('experiments')
    else:
        exp_dir = Path(exp_dir)
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Get experiment name
    exp_name = exp_cfg.get('name', 'experiment')
    
    # Create experiment subdirectory
    exp_path = exp_dir / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard logger
    if exp_cfg.get('create_tensorboard_logger', True):
        from lightning.pytorch.loggers import TensorBoardLogger
        tb_logger = TensorBoardLogger(
            save_dir=str(exp_dir),
            name=exp_name,
        )
        trainer.loggers.append(tb_logger)
    
    # Setup WandB logger
    if exp_cfg.get('create_wandb_logger', False):
        try:
            from lightning.pytorch.loggers import WandbLogger
            wandb_kwargs = exp_cfg.get('wandb_logger_kwargs', {})
            wandb_logger = WandbLogger(
                project=wandb_kwargs.get('project', 'experiment'),
                name=wandb_kwargs.get('name', exp_name),
                save_dir=str(exp_dir),
            )
            trainer.loggers.append(wandb_logger)
        except ImportError:
            logger.warning("wandb not installed, skipping WandB logger")
    
    # Setup checkpoint callback
    if exp_cfg.get('create_checkpoint_callback', True):
        from lightning.pytorch.callbacks import ModelCheckpoint, Callback
        
        checkpoint_params = exp_cfg.get('checkpoint_callback_params', {})
        monitor = checkpoint_params.get('monitor', 'val_loss')
        mode = checkpoint_params.get('mode', 'min')
        save_top_k = checkpoint_params.get('save_top_k', 1)
        filename = checkpoint_params.get('filename', '{epoch}-{step}')
        verbose = checkpoint_params.get('verbose', True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(exp_path / 'checkpoints'),
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=True,
            verbose=verbose,
        )
        trainer.callbacks.append(checkpoint_callback)
        
        # Add callback to log checkpoint save events (aligned with NeMo)
        # Only log from rank 0 to avoid duplicate output in DDP
        from utils.logging import is_global_rank_zero
        
        class CheckpointLoggingCallback(Callback):
            def on_save_checkpoint(self, trainer, pl_module, checkpoint):
                # Only log from rank 0 (aligned with NeMo)
                if not is_global_rank_zero():
                    return
                    
                if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback is not None:
                    cb = trainer.checkpoint_callback
                    if hasattr(cb, 'best_model_path') and cb.best_model_path:
                        best_score = getattr(cb, 'best_model_score', None)
                        if best_score is not None:
                            logger.info(
                                f"Best {monitor} checkpoint saved: {cb.best_model_path} "
                                f"(score: {best_score:.4f})"
                            )
        
        trainer.callbacks.append(CheckpointLoggingCallback())
    
    logger.info(f"Experiment directory: {exp_path}")

