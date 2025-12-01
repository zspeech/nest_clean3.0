#!/usr/bin/env python3
"""
NeMo training script with output saver.

This is a wrapper script that can be used in NeMo environment to run training
and save outputs for comparison.

Usage:
    python tools/nemo_training_with_saver.py \
        --config-path <config_path> \
        --config-name <config_name> \
        output_dir=./saved_nemo_outputs \
        seed=42 \
        save_steps="0,1,2,3,4,10,20,50" \
        <other training args>

Note: Use Hydra syntax (key=value) instead of --key value for output_dir, seed, save_steps
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np

try:
    import lightning.pytorch as pl
    from omegaconf import OmegaConf
    import nemo.collections.asr as nemo_asr
    from nemo.core.config import hydra_runner
    from nemo.utils import logging
    from nemo.utils.exp_manager import exp_manager
except ImportError:
    raise ImportError("NeMo is required. Please run this script in NeMo environment.")

# Import training output saver
# Note: This assumes the saver module is available in the path
# You may need to copy training_output_saver.py to NeMo environment or adjust import
try:
    from tools.training_output_saver import TrainingOutputSaver
except ImportError:
    # Fallback: try importing from nest_ssl_project if available
    import sys
    nest_ssl_path = Path(__file__).parent.parent.parent / 'nest_ssl_project'
    if nest_ssl_path.exists():
        sys.path.insert(0, str(nest_ssl_path))
        from tools.training_output_saver import TrainingOutputSaver
    else:
        raise ImportError("Cannot import TrainingOutputSaver. Please ensure training_output_saver.py is available.")


def set_seed(seed: int):
    """Set random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainingOutputSaverCallback(pl.Callback):
    """PyTorch Lightning callback to save training outputs."""
    
    def __init__(self, output_dir: str, seed: int, save_steps: list = None):
        """
        Initialize callback.
        
        Args:
            output_dir: Directory to save outputs
            seed: Random seed
            save_steps: List of step numbers to save (None = save all)
        """
        self.output_dir = output_dir
        self.seed = seed
        self.save_steps = set(save_steps) if save_steps else None
        
        self.saver = None
        self.current_step = 0
    
    def on_train_start(self, trainer, pl_module):
        """Setup saver when training starts."""
        set_seed(self.seed)
        
        self.saver = TrainingOutputSaver(
            output_dir=self.output_dir,
            seed=self.seed,
            save_every_n_steps=1 if self.save_steps is None else 999999,  # Only save specified steps
        )
        self.saver.setup_hooks(pl_module)
        self.saver.save_model_structure(pl_module)
        
        logging.info(f"TrainingOutputSaver initialized. Output dir: {self.output_dir}, Seed: {self.seed}")
        if self.save_steps:
            logging.info(f"Will save outputs for steps: {sorted(self.save_steps)}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Save outputs after each training batch."""
        self.current_step = trainer.global_step
        
        # Check if we should save this step
        if self.save_steps is not None and self.current_step not in self.save_steps:
            return
        
        # Get forward output and loss from outputs
        if isinstance(outputs, dict):
            forward_output = outputs.get('logprobs')  # Adjust based on your model
            loss = outputs.get('loss')
        else:
            # Try to get from model's last output
            forward_output = None
            loss = outputs if isinstance(outputs, torch.Tensor) else None
        
        if loss is not None:
            self.saver.save_step(
                step=self.current_step,
                batch=batch,
                forward_output=forward_output,
                loss=loss,
            )
    
    def on_train_end(self, trainer, pl_module):
        """Finalize saver when training ends."""
        if self.saver:
            self.saver.finalize()
            self.saver.cleanup()


def parse_save_steps(save_steps_str: str) -> list:
    """Parse save_steps string like '0,1,2,3,4,10,20,50' into list."""
    if not save_steps_str:
        return None
    return [int(x.strip()) for x in save_steps_str.split(',')]


@hydra_runner(config_path="../conf/ssl/nest", config_name="nest_fast-conformer")
def main(cfg):
    # Get parameters from Hydra config or use defaults
    # These can be set via command line: output_dir=./saved_nemo_outputs seed=42 save_steps="0,1,2,3,4"
    output_dir = cfg.get('output_dir', './saved_nemo_outputs')
    seed = cfg.get('seed', 42)
    save_steps_str = cfg.get('save_steps', None)
    save_steps = parse_save_steps(save_steps_str) if save_steps_str else None
    
    # Validate required parameters
    if output_dir is None:
        raise ValueError("output_dir must be specified. Use: output_dir=./saved_nemo_outputs")
    
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Seed: {seed}")
    logging.info(f"Save steps: {save_steps}")
    
    # Set seed
    set_seed(seed)
    
    # Create trainer
    trainer = pl.Trainer(**cfg.trainer)
    
    # Add output saver callback
    saver_callback = TrainingOutputSaverCallback(
        output_dir=output_dir,
        seed=seed,
        save_steps=save_steps,
    )
    trainer.callbacks.append(saver_callback)
    
    # Setup exp manager
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Create model
    asr_model = nemo_asr.models.EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=trainer)
    
    # Initialize from pretrained if specified
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)
    
    # Train
    trainer.fit(asr_model)


if __name__ == "__main__":
    main()

