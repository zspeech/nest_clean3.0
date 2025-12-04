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
Pure PyTorch SSL Model - No Lightning dependency, single GPU version.

This is a standalone implementation that:
- Uses only PyTorch (no Lightning)
- Single GPU training
- All initialization logic from original model
- Simple training loop interface
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.ssl_modules.masking import ConvFeatureMaksingWrapper
from core.classes.serialization import Serialization
from utils.logging import get_logger

__all__ = ['PureTorchSSLModel']

logger = get_logger(__name__)


class PureTorchSSLModel(nn.Module):
    """
    Pure PyTorch SSL Model for denoising and masked token prediction.
    
    No PyTorch Lightning dependency. Single GPU training.
    
    Usage:
        model = PureTorchSSLModel(cfg)
        model.to('cuda')
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        for batch in dataloader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize pure PyTorch SSL model.
        
        Args:
            cfg: Model configuration containing:
                - preprocessor: Audio preprocessor config
                - encoder: Conformer encoder config
                - decoder: Token prediction decoder config
                - quantizer: Audio quantizer config
                - masking: Masking strategy config
                - loss: Loss function config
                - mask_position: "pre_conv" or "post_conv" (default: "pre_conv")
        """
        super().__init__()
        
        # Store config
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self._cfg = cfg
        
        # Handle mask_position config adjustment BEFORE creating components
        if self._cfg.get("mask_position", "pre_conv") == "post_conv":
            with open_dict(self._cfg):
                self._cfg.quantizer.feat_in = self._cfg.encoder.d_model
                self._cfg.masking.feat_in = self._cfg.encoder.d_model
                self._cfg.masking.block_size = self._cfg.masking.block_size // self._cfg.encoder.subsampling_factor
                self._cfg.loss.combine_time_steps = 1
        
        # Initialize all components (same order as original)
        logger.info("Initializing preprocessor...")
        self.preprocessor = Serialization.from_config_dict(self._cfg.preprocessor)
        
        logger.info("Initializing quantizer...")
        self.quantizer = Serialization.from_config_dict(self._cfg.quantizer)
        
        logger.info("Initializing mask_processor...")
        self.mask_processor = Serialization.from_config_dict(self._cfg.masking)
        
        logger.info("Initializing encoder...")
        self.encoder = Serialization.from_config_dict(self._cfg.encoder)
        
        logger.info("Initializing decoder...")
        self.decoder = Serialization.from_config_dict(self._cfg.decoder)
        
        logger.info("Initializing loss...")
        self.loss = Serialization.from_config_dict(self._cfg.loss)
        
        # Handle post-conv masking wrapper
        self.pre_encoder = None
        if self._cfg.get("mask_position", "pre_conv") == "post_conv":
            logger.info("Setting up post-conv masking wrapper...")
            self.pre_encoder = ConvFeatureMaksingWrapper(self.encoder.pre_encode, self.mask_processor)
            self.encoder.pre_encode = self.pre_encoder
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    @property
    def cfg(self) -> DictConfig:
        """Get model configuration."""
        return self._cfg
    
    def forward(
        self,
        input_signal: Optional[torch.Tensor] = None,
        input_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
        noisy_input_signal: Optional[torch.Tensor] = None,
        noisy_input_signal_length: Optional[torch.Tensor] = None,
        processed_noisy_signal: Optional[torch.Tensor] = None,
        processed_noisy_signal_length: Optional[torch.Tensor] = None,
        apply_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_signal: Clean audio [B, T]
            input_signal_length: Clean audio lengths [B]
            processed_signal: Preprocessed clean features [B, D, T]
            processed_signal_length: Preprocessed clean lengths [B]
            noisy_input_signal: Noisy audio [B, T]
            noisy_input_signal_length: Noisy audio lengths [B]
            processed_noisy_signal: Preprocessed noisy features [B, D, T]
            processed_noisy_signal_length: Preprocessed noisy lengths [B]
            apply_mask: Whether to apply masking
        
        Returns:
            tuple: (log_probs, encoded_len, masks, tokens)
        """
        # Process clean signal
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        
        if not (has_input_signal ^ has_processed_signal):
            raise ValueError(
                "Either (input_signal, input_signal_length) or "
                "(processed_signal, processed_signal_length) must be provided, but not both."
            )
        
        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        
        # Process noisy signal
        has_noisy_input = noisy_input_signal is not None and noisy_input_signal_length is not None
        has_processed_noisy = processed_noisy_signal is not None and processed_noisy_signal_length is not None
        
        if not (has_noisy_input ^ has_processed_noisy):
            raise ValueError(
                "Either (noisy_input_signal, noisy_input_signal_length) or "
                "(processed_noisy_signal, processed_noisy_signal_length) must be provided, but not both."
            )
        
        if not has_processed_noisy:
            processed_noisy_signal, processed_noisy_signal_length = self.preprocessor(
                input_signal=noisy_input_signal,
                length=noisy_input_signal_length,
            )
        
        # Core forward logic
        if self.pre_encoder is not None:
            # Post-conv masking
            feats, _ = self.pre_encoder.pre_encode(x=processed_signal, lengths=processed_signal_length)
            _, tokens = self.quantizer(input_signal=feats.transpose(1, 2))
            self.pre_encoder.set_masking_enabled(apply_mask=apply_mask)
            
            encoded, encoded_len = self.encoder(
                audio_signal=processed_noisy_signal, length=processed_noisy_signal_length
            )
            masks = self.pre_encoder.get_current_mask()
        else:
            # Pre-conv masking
            _, tokens = self.quantizer(input_signal=processed_signal)
            
            if apply_mask:
                masked_signal, masks = self.mask_processor(
                    input_feats=processed_noisy_signal, input_lengths=processed_noisy_signal_length
                )
            else:
                masked_signal = processed_noisy_signal
                masks = torch.zeros_like(processed_noisy_signal)
            
            encoded, encoded_len = self.encoder(
                audio_signal=masked_signal, length=processed_noisy_signal_length
            )
        
        # Decode
        log_probs = self.decoder(encoder_output=encoded)
        
        return log_probs, encoded_len, masks, tokens
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        encoded_len: torch.Tensor,
        masks: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss from forward outputs.
        
        Args:
            log_probs: Decoder log probabilities
            encoded_len: Encoded sequence lengths
            masks: Applied masks
            tokens: Target tokens
        
        Returns:
            Loss value
        """
        return self.loss(
            masks=masks,
            decoder_outputs=log_probs,
            targets=tokens,
            decoder_lengths=encoded_len,
        )
    
    def training_step(self, batch) -> torch.Tensor:
        """
        Single training step.
        
        Args:
            batch: Can be:
                - AudioNoiseBatch object with .audio, .audio_len, .noisy_audio, .noisy_audio_len
                - Tuple: (audio, audio_len, noise, noise_len, noisy_audio, noisy_audio_len)
                - Dict with keys: 'audio', 'audio_len', 'noisy_audio', 'noisy_audio_len'
        
        Returns:
            Loss tensor (call .backward() on it)
        """
        # Extract data from batch
        if hasattr(batch, 'audio'):
            # AudioNoiseBatch object
            audio = batch.audio
            audio_len = batch.audio_len
            noisy_audio = batch.noisy_audio
            noisy_audio_len = batch.noisy_audio_len
        elif isinstance(batch, (tuple, list)) and len(batch) >= 6:
            # Tuple format
            audio = batch[0]
            audio_len = batch[1]
            noisy_audio = batch[4]
            noisy_audio_len = batch[5]
        elif isinstance(batch, dict):
            # Dict format
            audio = batch['audio']
            audio_len = batch['audio_len']
            noisy_audio = batch['noisy_audio']
            noisy_audio_len = batch['noisy_audio_len']
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
        
        # Forward pass
        log_probs, encoded_len, masks, tokens = self.forward(
            input_signal=audio,
            input_signal_length=audio_len,
            noisy_input_signal=noisy_audio,
            noisy_input_signal_length=noisy_audio_len,
            apply_mask=True,
        )
        
        # Compute loss
        loss = self.compute_loss(log_probs, encoded_len, masks, tokens)
        
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch) -> torch.Tensor:
        """
        Single validation step (no gradients).
        
        Args:
            batch: Same format as training_step
        
        Returns:
            Loss tensor
        """
        # Extract data from batch
        if hasattr(batch, 'audio'):
            audio = batch.audio
            audio_len = batch.audio_len
            noisy_audio = batch.noisy_audio
            noisy_audio_len = batch.noisy_audio_len
        elif isinstance(batch, (tuple, list)) and len(batch) >= 6:
            audio = batch[0]
            audio_len = batch[1]
            noisy_audio = batch[4]
            noisy_audio_len = batch[5]
        elif isinstance(batch, dict):
            audio = batch['audio']
            audio_len = batch['audio_len']
            noisy_audio = batch['noisy_audio']
            noisy_audio_len = batch['noisy_audio_len']
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
        
        # Forward pass
        log_probs, encoded_len, masks, tokens = self.forward(
            input_signal=audio,
            input_signal_length=audio_len,
            noisy_input_signal=noisy_audio,
            noisy_input_signal_length=noisy_audio_len,
            apply_mask=True,
        )
        
        # Compute loss
        loss = self.compute_loss(log_probs, encoded_len, masks, tokens)
        
        return loss
    
    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, step: int = 0):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer to save state
            epoch: Current epoch number
            step: Current step number
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': OmegaConf.to_container(self._cfg, resolve=True),
            'epoch': epoch,
            'step': step,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str, optimizer=None, strict: bool = True):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            optimizer: Optional optimizer to load state into
            strict: Whether to strictly enforce state_dict keys match
        
        Returns:
            Dict with 'epoch' and 'step' if available
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        logger.info(f"Loaded model state from {path}")
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
        }
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'PureTorchSSLModel':
        """
        Create model from config file.
        
        Args:
            config_path: Path to YAML config file
        
        Returns:
            PureTorchSSLModel instance
        """
        cfg = OmegaConf.load(config_path)
        return cls(cfg.model if 'model' in cfg else cfg)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> 'PureTorchSSLModel':
        """
        Create model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            PureTorchSSLModel instance
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg = OmegaConf.create(checkpoint['config'])
        model = cls(cfg)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def create_optimizer(model: PureTorchSSLModel, cfg: DictConfig) -> torch.optim.Optimizer:
    """
    Create optimizer from config.
    
    Args:
        model: Model to optimize
        cfg: Optimizer config with 'name', 'lr', 'weight_decay', etc.
    
    Returns:
        Optimizer instance
    """
    optim_name = cfg.get('name', 'adamw').lower()
    lr = cfg.get('lr', 1e-4)
    weight_decay = cfg.get('weight_decay', 0.0)
    betas = cfg.get('betas', [0.9, 0.999])
    
    if optim_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )
    elif optim_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )
    elif optim_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg.get('momentum', 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")


def create_scheduler(optimizer, cfg: DictConfig, num_training_steps: int = None):
    """
    Create learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer instance
        cfg: Scheduler config
        num_training_steps: Total training steps (for some schedulers)
    
    Returns:
        Scheduler instance or None
    """
    if cfg is None or not cfg.get('name'):
        return None
    
    sched_name = cfg.name.lower()
    
    if sched_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps or cfg.get('t_max', 10000),
            eta_min=cfg.get('min_lr', 0.0),
        )
    elif sched_name == 'warmup_cosine':
        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = cfg.get('warmup_steps', 1000)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 0.5 * (1 + torch.cos(torch.tensor(
                (step - warmup_steps) / (num_training_steps - warmup_steps) * 3.14159
            )).item())
        
        return LambdaLR(optimizer, lr_lambda)
    elif sched_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.get('step_size', 1000),
            gamma=cfg.get('gamma', 0.1),
        )
    else:
        logger.warning(f"Unknown scheduler: {sched_name}, returning None")
        return None

