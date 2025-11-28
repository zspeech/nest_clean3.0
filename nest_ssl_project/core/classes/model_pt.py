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
Simplified ModelPT base class to replace nemo.core.classes.ModelPT
This provides the essential functionality for PyTorch Lightning-based models.
"""

import os
import tarfile
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf, open_dict

from core.classes.neural_module import NeuralModule
from utils.logging import get_logger

logger = get_logger(__name__)


class ModelPT(pl.LightningModule, NeuralModule, ABC):
    """
    Base class for PyTorch Lightning-based models.
    Simplified version of NeMo's ModelPT with essential functionality.
    """
    
    def __init__(self, cfg: DictConfig, trainer: Optional[pl.Trainer] = None):
        """
        Initialize the model.
        
        Args:
            cfg: Configuration object (DictConfig)
            trainer: Optional PyTorch Lightning Trainer instance
        """
        if trainer is not None and not isinstance(trainer, pl.Trainer):
            raise ValueError(
                f"trainer must be None or lightning.pytorch.Trainer, got {type(trainer)}"
            )
        
        super().__init__()
        
        # Convert config to DictConfig if needed
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        
        # Store config
        self._cfg = cfg
        self.save_hyperparameters("cfg")
        
        # Data loaders
        self._train_dl = None
        self._validation_dl = None
        self._test_dl = None
        
        # Optimizer and scheduler
        self._optimizer = None
        self._scheduler = None
        
        # Set trainer
        self.set_trainer(trainer)
        
        # Validation/test outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Setup data loaders if config provided
        if self._cfg is not None:
            if 'train_ds' in self._cfg and self._cfg.train_ds is not None:
                if not self._cfg.train_ds.get('defer_setup', False):
                    self.setup_training_data(self._cfg.train_ds)
            
            if 'validation_ds' in self._cfg and self._cfg.validation_ds is not None:
                if not self._cfg.validation_ds.get('defer_setup', False):
                    self.setup_validation_data(self._cfg.validation_ds)
            
            if 'test_ds' in self._cfg and self._cfg.test_ds is not None:
                if not self._cfg.test_ds.get('defer_setup', False):
                    self.setup_test_data(self._cfg.test_ds)
    
    @property
    def cfg(self) -> DictConfig:
        """Get model configuration."""
        return self._cfg
    
    def set_trainer(self, trainer: Optional[pl.Trainer]):
        """Set the trainer instance."""
        self._trainer = trainer

    def _update_dataset_config(self, dataset_name: str, config: Optional[Union[DictConfig, Dict]]):
        """
        Update the config (if not None) of the dataset by given name.

        Simplified version of NeMo's ModelPT._update_dataset_config.

        Args:
            dataset_name: 'train', 'validation' or 'test'.
            config: Optional DictConfig or dict. If None is passed, this method simply returns.
        """
        if config is None:
            return

        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)

        if dataset_name not in ['train', 'validation', 'test']:
            raise ValueError("`dataset_name` must be one of ['train', 'validation', 'test']")

        key_name = dataset_name + "_ds"

        # Temporarily disable struct to allow assignment
        OmegaConf.set_struct(self._cfg, False)
        self._cfg[key_name] = config
        OmegaConf.set_struct(self._cfg, True)

    
    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]):
        """
        Instantiate a module from a configuration dictionary.
        Supports Hydra-style instantiation with _target_ key.
        Uses Serialization.from_config_dict for remapping NeMo targets.
        """
        from core.classes.serialization import Serialization
        return Serialization.from_config_dict(config)
    
    @abstractmethod
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """Setup training data loader."""
        pass
    
    @abstractmethod
    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """Setup validation data loader."""
        pass
    
    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """Setup test data loader (optional)."""
        pass
    
    def train_dataloader(self):
        """Return training dataloader."""
        return self._train_dl
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return self._validation_dl
    
    def test_dataloader(self):
        """Return test dataloader."""
        return self._test_dl
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler from config."""
        if 'optim' not in self._cfg:
            return None
        
        optim_cfg = self._cfg.optim
        
        # Get optimizer
        optim_name = optim_cfg.get('name', 'adam')
        lr = optim_cfg.get('lr', 1e-3)
        
        if optim_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                betas=optim_cfg.get('betas', [0.9, 0.999]),
                weight_decay=optim_cfg.get('weight_decay', 0.0)
            )
        elif optim_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                betas=optim_cfg.get('betas', [0.9, 0.999]),
                weight_decay=optim_cfg.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optim_name}")
        
        self._optimizer = optimizer
        
        # Get scheduler if specified
        if 'sched' in optim_cfg and optim_cfg.sched is not None:
            sched_cfg = optim_cfg.sched
            sched_name = sched_cfg.get('name', 'constant')
            
            if sched_name == 'NoamAnnealing':
                # Noam annealing scheduler
                d_model = sched_cfg.get('d_model', 512)
                warmup_steps = sched_cfg.get('warmup_steps', 4000)
                
                def lr_lambda(step):
                    step = max(1, step)
                    return min(step ** (-0.5), step * warmup_steps ** (-1.5)) * (d_model ** (-0.5))
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            elif sched_name == 'CosineAnnealing':
                T_max = sched_cfg.get('T_max', 100)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            else:
                # Default: constant learning rate
                scheduler = None
            
            if scheduler is not None:
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                    }
                }
        
        return optimizer
    
    def save_to(self, save_path: str):
        """
        Save model to .nemo file (tar archive).
        Simplified version - saves config and checkpoint.
        """
        save_path = Path(save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save config
            config_path = Path(tmpdir) / 'model_config.yaml'
            with open(config_path, 'w') as f:
                OmegaConf.save(self._cfg, f)
            
            # Save checkpoint
            ckpt_path = Path(tmpdir) / 'model_weights.ckpt'
            torch.save(self.state_dict(), ckpt_path)
            
            # Create tar archive
            with tarfile.open(save_path, 'w:gz') as tar:
                tar.add(config_path, arcname='model_config.yaml')
                tar.add(ckpt_path, arcname='model_weights.ckpt')
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[str] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
    ):
        """
        Restore model from .nemo file or checkpoint.
        """
        restore_path = Path(restore_path).expanduser().resolve()
        
        if restore_path.suffix == '.nemo':
            # Load from .nemo archive
            with tempfile.TemporaryDirectory() as tmpdir:
                with tarfile.open(restore_path, 'r:gz') as tar:
                    tar.extractall(tmpdir)
                
                # Load config
                config_path = Path(tmpdir) / 'model_config.yaml'
                if override_config_path:
                    config_path = Path(override_config_path)
                
                cfg = OmegaConf.load(config_path)
                
                # Create model instance
                instance = cls(cfg=cfg, trainer=None)
                
                # Load weights
                ckpt_path = Path(tmpdir) / 'model_weights.ckpt'
                state_dict = torch.load(ckpt_path, map_location=map_location)
                instance.load_state_dict(state_dict, strict=strict)
                
                return instance
        else:
            # Load from checkpoint
            checkpoint = torch.load(restore_path, map_location=map_location)
            
            if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
                cfg = checkpoint['hyper_parameters']['cfg']
                if override_config_path:
                    cfg = OmegaConf.load(override_config_path)
            else:
                raise ValueError("Checkpoint does not contain config. Please provide override_config_path.")
            
            instance = cls(cfg=cfg, trainer=None)
            instance.load_state_dict(checkpoint['state_dict'], strict=strict)
            
            return instance
    
    def maybe_init_from_pretrained_checkpoint(self, cfg: DictConfig):
        """
        Initialize model from pretrained checkpoint if specified in config.
        """
        if 'init_from_pretrained_model' in cfg:
            pretrained_path = cfg.init_from_pretrained_model
            logger.info(f"Loading pretrained model from {pretrained_path}")
            
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out incompatible keys
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            logger.info(f"Loaded {len(pretrained_dict)} parameters from pretrained model")

