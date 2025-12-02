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


import lightning.pytorch as pl
from omegaconf import OmegaConf
import torch
import numpy as np
from pathlib import Path

# Import local utilities
from utils.hydra_runner import hydra_runner
from utils.logging import get_logger
from utils.exp_manager import exp_manager

# Import local model
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ssl_models import EncDecDenoiseMaskedTokenPredModel
from tools.training_output_saver import TrainingOutputSaver


"""
# Example of training a self-supervised denoising masked token prediction model with output saving
```sh
python train_with_saver.py \
    # (Optional: --config-path=config --config-name=nest_fast-conformer) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.train_ds.noise_manifest=<path to noise manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.validation_ds.noise_manifest=<path to noise manifest> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    strategy="ddp"  \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000 \
    output_dir=./saved_nest_outputs \
    seed=42 \
    save_steps="0"
```
"""


logger = get_logger(__name__)


def set_seed(seed: int):
    """Set random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_save_steps(save_steps_str: str) -> list:
    """Parse save_steps string like '0,1,2,3,4,10,20,50' into list."""
    if not save_steps_str:
        return None
    # Remove quotes if present and split by comma
    save_steps_str = save_steps_str.strip().strip('"').strip("'")
    if not save_steps_str:
        return None
    return [int(x.strip()) for x in save_steps_str.split(',')]


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
        self.weights_saved = False  # Track if weights have been saved for step 0
    
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
        self.saver.save_buffers(pl_module)
        
        # Register hook on decoder to capture forward output (log_probs)
        def decoder_hook(module, input, output):
            pl_module._last_forward_output = output[0] if isinstance(output, tuple) else output
        
        if hasattr(pl_module, 'decoder'):
            self.decoder_hook_handle = pl_module.decoder.register_forward_hook(decoder_hook)
        else:
            self.decoder_hook_handle = None
        
        from utils.logging import is_global_rank_zero
        if is_global_rank_zero():
            logger.info(f"TrainingOutputSaver initialized. Output dir: {self.output_dir}, Seed: {self.seed}")
            if self.save_steps:
                logger.info(f"Will save outputs for steps: {sorted(self.save_steps)}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Save outputs after each training batch."""
        self.current_step = trainer.global_step
        
        from utils.logging import is_global_rank_zero
        if is_global_rank_zero():
            logger.info(f"on_train_batch_end: global_step={self.current_step}, batch_idx={batch_idx}, save_steps={self.save_steps}")
        
        # Check if we should save this step
        # If save_steps is specified, check if current step is in the list
        # Otherwise, save the first batch (batch_idx == 0)
        should_save = False
        if self.save_steps is not None:
            if self.current_step in self.save_steps:
                should_save = True
                if is_global_rank_zero():
                    logger.info(f"Step {self.current_step} is in save_steps list")
            else:
                if is_global_rank_zero():
                    logger.info(f"Skipping step {self.current_step} (not in save_steps: {self.save_steps})")
        else:
            # If no save_steps specified, save first batch
            if batch_idx == 0:
                should_save = True
                if is_global_rank_zero():
                    logger.info(f"Saving first batch (batch_idx=0)")
        
        if not should_save:
            return
        
        if is_global_rank_zero():
            logger.info(f"Saving outputs for step {self.current_step} (batch_idx={batch_idx})")
        
        # Get forward output and loss from outputs
        # For EncDecDenoiseMaskedTokenPredModel, outputs is a dict with 'loss' and 'log'
        if isinstance(outputs, dict):
            loss = outputs.get('loss')
        else:
            loss = outputs if isinstance(outputs, torch.Tensor) else None
        
        # Try to get forward output from model's stored attribute (set in training_step hook)
        # Or try to get from decoder output hook
        forward_output = getattr(pl_module, '_last_forward_output', None)
        
        if loss is not None:
            # Save forward outputs and gradients
            # Use force_save=True to bypass save_every_n_steps check since we already checked save_steps
            self.saver.save_step(
                step=self.current_step,
                batch=batch,
                forward_output=forward_output,
                loss=loss,
                save_batch=True,
                save_weights=False,  # Save weights separately after optimizer step
                force_save=True,  # Force save since we already checked save_steps
            )
            from utils.logging import is_global_rank_zero
            if is_global_rank_zero():
                logger.info(f"Successfully saved outputs for step {self.current_step}")
        else:
            from utils.logging import is_global_rank_zero
            if is_global_rank_zero():
                logger.warning(f"Loss is None for step {self.current_step}, skipping save")
            
            # Save model weights after optimizer.step() (which happens before on_train_batch_end)
            if not self.weights_saved:
                step_dir = self.saver.output_dir / f"step_{self.current_step}"
                if step_dir.exists():
                    param_weights = {}
                    for name, param in pl_module.named_parameters():
                        param_weights[name] = param.detach().cpu().clone()
                    
                    torch.save(param_weights, step_dir / 'parameter_weights.pt')
                    from utils.logging import is_global_rank_zero
                    if is_global_rank_zero():
                        logger.info(f"Saved model weights after optimizer step for step {self.current_step}")
                    self.weights_saved = True
    
    def on_train_end(self, trainer, pl_module):
        """Finalize saver when training ends."""
        if self.saver:
            self.saver.finalize()
            self.saver.cleanup()
        
        # Remove decoder hook
        if hasattr(self, 'decoder_hook_handle') and self.decoder_hook_handle is not None:
            self.decoder_hook_handle.remove()


@hydra_runner(config_path="config", config_name="nest_fast-conformer")
def main(cfg):
    # Only print from rank 0 in DDP mode to avoid duplicate output
    from utils.logging import is_global_rank_zero
    
    # Get parameters from Hydra config
    output_dir = cfg.get('output_dir', './saved_nest_outputs')
    seed = cfg.get('seed', 42)
    save_steps_str = cfg.get('save_steps', None)
    save_steps = parse_save_steps(save_steps_str) if save_steps_str else None
    
    if is_global_rank_zero():
        logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Starting training with {cfg.trainer.get('devices', 1)} device(s)")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Save steps: {save_steps}")
    
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
    asr_model = EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=trainer)
    
    # Initialize from pretrained if specified
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)
    
    # Load weights from NeMo output if specified (for alignment testing)
    nemo_weights_path = cfg.get('load_nemo_weights', None)
    if nemo_weights_path:
        nemo_weights_path = Path(nemo_weights_path)
        if nemo_weights_path.exists():
            if is_global_rank_zero():
                logger.info(f"Loading weights from NeMo output: {nemo_weights_path}")
            nemo_weights = torch.load(nemo_weights_path, map_location='cpu', weights_only=False)
            
            # Load weights into model
            model_state_dict = asr_model.state_dict()
            loaded_count = 0
            for name, param in nemo_weights.items():
                if name in model_state_dict:
                    if model_state_dict[name].shape == param.shape:
                        model_state_dict[name] = param
                        loaded_count += 1
                    else:
                        if is_global_rank_zero():
                            logger.warning(f"Shape mismatch for {name}: {model_state_dict[name].shape} vs {param.shape}")
            
            asr_model.load_state_dict(model_state_dict)
            if is_global_rank_zero():
                logger.info(f"Loaded {loaded_count}/{len(nemo_weights)} weights from NeMo")
        else:
            if is_global_rank_zero():
                logger.warning(f"NeMo weights path does not exist: {nemo_weights_path}")
    
    # Log training start from rank 0 only
    if is_global_rank_zero():
        logger.info(f"Training started. World size: {trainer.world_size}, Global rank: {trainer.global_rank}")
    
    # Train
    trainer.fit(asr_model)


if __name__ == "__main__":
    main()

