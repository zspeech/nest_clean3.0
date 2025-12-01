#!/usr/bin/env python3
"""
nest_ssl_project training script with output comparator.

This script runs training and compares outputs with saved NeMo outputs.

Usage:
    python tools/nest_training_with_comparator.py \
        --config-path config \
        --config-name nest_fast-conformer \
        --saved_outputs_dir ./saved_nemo_outputs \
        --comparison_output_dir ./comparison_results \
        --seed 42 \
        <other training args>
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np

import lightning.pytorch as pl
from omegaconf import OmegaConf

# Import local utilities
from utils.hydra_runner import hydra_runner
from utils.logging import get_logger
from utils.exp_manager import exp_manager

# Import local model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ssl_models import EncDecDenoiseMaskedTokenPredModel
from tools.training_output_saver import TrainingOutputComparator, ForwardBackwardHook

logger = get_logger(__name__)


def set_seed(seed: int):
    """Set random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainingOutputComparatorCallback(pl.Callback):
    """PyTorch Lightning callback to compare training outputs."""
    
    def __init__(
        self,
        saved_outputs_dir: str,
        comparison_output_dir: str = None,
        seed: int = 42,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        """
        Initialize callback.
        
        Args:
            saved_outputs_dir: Directory containing saved NeMo outputs
            comparison_output_dir: Directory to save comparison results
            seed: Random seed (should match NeMo training)
            atol: Absolute tolerance
            rtol: Relative tolerance
        """
        self.saved_outputs_dir = saved_outputs_dir
        self.comparison_output_dir = comparison_output_dir
        self.seed = seed
        self.atol = atol
        self.rtol = rtol
        
        self.comparator = None
        self.hooks = {}
        self.current_step = 0
    
    def on_train_start(self, trainer, pl_module):
        """Setup comparator when training starts."""
        set_seed(self.seed)
        
        self.comparator = TrainingOutputComparator(
            saved_outputs_dir=self.saved_outputs_dir,
            comparison_output_dir=self.comparison_output_dir,
            atol=self.atol,
            rtol=self.rtol,
        )
        
        # Register hooks
        for name, module in pl_module.named_modules():
            if name == "":
                continue
            if name not in self.hooks:
                self.hooks[name] = ForwardBackwardHook(name)
            self.hooks[name].register(module)
        
        logger.info(f"TrainingOutputComparator initialized. Saved outputs dir: {self.saved_outputs_dir}")
        logger.info(f"Seed: {self.seed}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Compare outputs after each training batch."""
        self.current_step = trainer.global_step
        
        # Get forward output and loss from outputs
        if isinstance(outputs, dict):
            forward_output = outputs.get('logprobs')  # Adjust based on your model
            loss = outputs.get('loss')
        else:
            forward_output = None
            loss = outputs if isinstance(outputs, torch.Tensor) else None
        
        if loss is not None:
            # Compare with saved outputs
            comparison_result = self.comparator.compare_step(
                step=self.current_step,
                batch=batch,
                forward_output=forward_output,
                loss=loss,
                model=pl_module,
                hooks=self.hooks,
            )
            
            # Print comparison result
            if 'error' not in comparison_result:
                forward_match = comparison_result.get('forward_output_match', {}).get('match', False)
                loss_match = comparison_result.get('loss_match', {}).get('match', False)
                
                if is_global_rank_zero():
                    status = "✓" if (forward_match and loss_match) else "✗"
                    logger.info(
                        f"Step {self.current_step}: {status} "
                        f"Forward: {'✓' if forward_match else '✗'}, "
                        f"Loss: {'✓' if loss_match else '✗'}"
                    )
        
        # Clear hooks for next step
        for hook in self.hooks.values():
            hook.clear()
    
    def on_train_end(self, trainer, pl_module):
        """Print summary when training ends."""
        if self.comparator:
            if is_global_rank_zero():
                self.comparator.print_summary()
        
        # Cleanup hooks
        for hook in self.hooks.values():
            hook.remove()


def is_global_rank_zero():
    """Check if current process is global rank 0."""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


@hydra_runner(config_path="config", config_name="nest_fast-conformer")
def main(cfg):
    # Get parameters from Hydra config (can be set in config file or via command line override)
    saved_outputs_dir = cfg.get('saved_outputs_dir', './saved_nemo_outputs')
    comparison_output_dir = cfg.get('comparison_output_dir', None)
    seed = cfg.get('seed', 42)
    atol = cfg.get('atol', 1e-5)
    rtol = cfg.get('rtol', 1e-5)
    
    if is_global_rank_zero():
        logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Saved outputs directory: {saved_outputs_dir}")
        logger.info(f"Comparison output directory: {comparison_output_dir}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Tolerance: atol={atol}, rtol={rtol}")
    
    # Set seed
    set_seed(seed)
    
    # Create trainer
    trainer = pl.Trainer(**cfg.trainer)
    
    # Add comparator callback
    comparator_callback = TrainingOutputComparatorCallback(
        saved_outputs_dir=saved_outputs_dir,
        comparison_output_dir=comparison_output_dir,
        seed=seed,
        atol=atol,
        rtol=rtol,
    )
    trainer.callbacks.append(comparator_callback)
    
    # Setup exp manager
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Create model
    asr_model = EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=trainer)
    
    # Initialize from pretrained if specified
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)
    
    # Log training start
    if is_global_rank_zero():
        logger.info(f"Training started. World size: {trainer.world_size}, Global rank: {trainer.global_rank}")
    
    # Train
    trainer.fit(asr_model)


if __name__ == "__main__":
    main()

