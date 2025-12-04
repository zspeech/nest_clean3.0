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
Simplified SSL Model - A lightweight version of EncDecDenoiseMaskedTokenPredModel
that focuses on core functionality and is easier to integrate with other frameworks.

This module provides a simplified model class that:
- Removes complex data loader setup (users handle their own data loading)
- Removes AccessMixin and other complex mixins
- Simplifies training/validation steps
- Maintains PyTorch Lightning compatibility
- Keeps core forward pass logic intact
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from omegaconf import DictConfig

# Import from local modules
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data import ssl_dataset
from modules.ssl_modules.masking import ConvFeatureMaksingWrapper
from core.classes import ModelPT
from utils.logging import get_logger

__all__ = ['SimplifiedSSLModel']

logging = get_logger(__name__)


class SimplifiedSSLModel(LightningModule):
    """
    Simplified SSL Model for denoising and masked token prediction.
    
    This is a lightweight version that:
    - Only handles core forward pass and loss computation
    - Does not manage data loaders (users provide their own)
    - Removes complex mixins and access control
    - Maintains compatibility with PyTorch Lightning
    
    Usage:
        model = SimplifiedSSLModel(cfg=cfg.model)
        # Users handle their own data loading
        trainer = pl.Trainer(...)
        trainer.fit(model, train_dataloader, val_dataloader)
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize simplified SSL model.
        
        Args:
            cfg: Model configuration dict containing:
                - preprocessor: Preprocessor config
                - encoder: Encoder config
                - decoder: Decoder config
                - quantizer: Quantizer config
                - masking: Masking config
                - loss: Loss config
                - mask_position: "pre_conv" or "post_conv" (default: "pre_conv")
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Initialize components using ModelPT's from_config_dict
        self.preprocessor = ModelPT.from_config_dict(cfg.preprocessor)
        self.quantizer = ModelPT.from_config_dict(cfg.quantizer)
        self.mask_processor = ModelPT.from_config_dict(cfg.masking)
        self.encoder = ModelPT.from_config_dict(cfg.encoder)
        self.decoder = ModelPT.from_config_dict(cfg.decoder)
        self.loss = ModelPT.from_config_dict(cfg.loss)
        
        # Handle post-conv masking if needed
        self.pre_encoder = None
        if cfg.get("mask_position", "pre_conv") == "post_conv":
            # Adjust config for post-convolution masking
            cfg.quantizer.feat_in = cfg.encoder.d_model
            cfg.masking.feat_in = cfg.encoder.d_model
            cfg.masking.block_size = cfg.masking.block_size // cfg.encoder.subsampling_factor
            cfg.loss.combine_time_steps = 1
            
            # Wrap pre_encode with masking
            self.pre_encoder = ConvFeatureMaksingWrapper(self.encoder.pre_encode, self.mask_processor)
            self.encoder.pre_encode = self.pre_encoder
        
        # Initialize validation/test outputs lists
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
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
    ):
        """
        Forward pass of the model.
        
        Args:
            input_signal: Clean audio signal [B, T] (optional if processed_signal provided)
            input_signal_length: Lengths of clean audio [B] (optional if processed_signal provided)
            processed_signal: Preprocessed clean features [B, D, T] (optional if input_signal provided)
            processed_signal_length: Lengths of processed clean features [B] (optional if processed_signal provided)
            noisy_input_signal: Noisy audio signal [B, T] (optional if processed_noisy_signal provided)
            noisy_input_signal_length: Lengths of noisy audio [B] (optional if processed_noisy_signal provided)
            processed_noisy_signal: Preprocessed noisy features [B, D, T] (optional if noisy_input_signal provided)
            processed_noisy_signal_length: Lengths of processed noisy features [B] (optional if processed_noisy_signal provided)
            apply_mask: Whether to apply masking (default: False)
        
        Returns:
            tuple: (log_probs, encoded_len, masks, tokens)
                - log_probs: Decoder log probabilities [B, T, C] or [B, T, C, H]
                - encoded_len: Encoded sequence lengths [B]
                - masks: Applied masks [B, D, T]
                - tokens: Quantized tokens [B, T] or [B, T, H]
        """
        # Process clean signal if needed
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
        
        # Process noisy signal if needed
        has_noisy_input_signal = noisy_input_signal is not None and noisy_input_signal_length is not None
        has_processed_noisy_signal = (
            processed_noisy_signal is not None and processed_noisy_signal_length is not None
        )
        
        if not (has_noisy_input_signal ^ has_processed_noisy_signal):
            raise ValueError(
                "Either (noisy_input_signal, noisy_input_signal_length) or "
                "(processed_noisy_signal, processed_noisy_signal_length) must be provided, but not both."
            )
        
        if not has_processed_noisy_signal:
            processed_noisy_signal, processed_noisy_signal_length = self.preprocessor(
                input_signal=noisy_input_signal,
                length=noisy_input_signal_length,
            )
        
        # Generate tokens from clean signal
        if self.pre_encoder is not None:
            # Post-conv masking: get features after subsampling
            feats, _ = self.pre_encoder.pre_encode(x=processed_signal, lengths=processed_signal_length)
            _, tokens = self.quantizer(input_signal=feats.transpose(1, 2))
            self.pre_encoder.set_masking_enabled(apply_mask=apply_mask)
            
            # Encode noisy signal
            encoded, encoded_len = self.encoder(
                audio_signal=processed_noisy_signal, length=processed_noisy_signal_length
            )
            masks = self.pre_encoder.get_current_mask()
        else:
            # Pre-conv masking: quantize clean signal
            _, tokens = self.quantizer(input_signal=processed_signal)
            
            # Apply masking to noisy signal
            if apply_mask:
                masked_signal, masks = self.mask_processor(
                    input_feats=processed_noisy_signal, input_lengths=processed_noisy_signal_length
                )
            else:
                masked_signal = processed_noisy_signal
                masks = torch.zeros_like(processed_noisy_signal)
            
            # Encode masked noisy signal
            encoded, encoded_len = self.encoder(
                audio_signal=masked_signal, length=processed_noisy_signal_length
            )
        
        # Decode
        log_probs = self.decoder(encoder_output=encoded)
        
        return log_probs, encoded_len, masks, tokens
    
    def training_step(self, batch, batch_idx: int):
        """
        Training step.
        
        Args:
            batch: Batch data. Can be:
                - ssl_dataset.AudioNoiseBatch object
                - Tuple/list: (audio, audio_len, noise, noise_len, noisy_audio, noisy_audio_len)
            batch_idx: Batch index
        
        Returns:
            dict: {'loss': loss_value, 'log': tensorboard_logs}
        """
        # Handle different batch formats
        if isinstance(batch, ssl_dataset.AudioNoiseBatch):
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=batch.audio,
                input_signal_length=batch.audio_len,
                noisy_input_signal=batch.noisy_audio,
                noisy_input_signal_length=batch.noisy_audio_len,
                apply_mask=True,
            )
        elif isinstance(batch, (tuple, list)) and len(batch) >= 6:
            # Assume format: (audio, audio_len, noise, noise_len, noisy_audio, noisy_audio_len)
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=batch[0],
                input_signal_length=batch[1],
                noisy_input_signal=batch[4],
                noisy_input_signal_length=batch[5],
                apply_mask=True,
            )
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
        
        # Compute loss
        loss_value = self.loss(
            masks=masks,
            decoder_outputs=log_probs,
            targets=tokens,
            decoder_lengths=encoded_len,
        )
        
        # Logging
        self.log('train_loss', loss_value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Get learning rate if optimizer is configured
        lr = 0.0
        if self.optimizers() is not None:
            opt = self.optimizers()
            if isinstance(opt, list):
                opt = opt[0]
            if hasattr(opt, 'param_groups') and len(opt.param_groups) > 0:
                lr = opt.param_groups[0].get('lr', 0.0)
        
        tensorboard_logs = {
            'learning_rate': lr,
            'global_step': self.global_step,
            'train_loss': loss_value,
        }
        
        return {'loss': loss_value, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """
        Validation step.
        
        Args:
            batch: Batch data (same format as training_step)
            batch_idx: Batch index
            dataloader_idx: Dataloader index (for multiple validation dataloaders)
        
        Returns:
            dict: {'val_loss': loss_value}
        """
        # Handle different batch formats
        if isinstance(batch, ssl_dataset.AudioNoiseBatch):
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=batch.audio,
                input_signal_length=batch.audio_len,
                noisy_input_signal=batch.noisy_audio,
                noisy_input_signal_length=batch.noisy_audio_len,
                apply_mask=True,
            )
        elif isinstance(batch, (tuple, list)) and len(batch) >= 6:
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=batch[0],
                input_signal_length=batch[1],
                noisy_input_signal=batch[4],
                noisy_input_signal_length=batch[5],
                apply_mask=True,
            )
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
        
        # Compute loss
        loss_value = self.loss(
            masks=masks,
            decoder_outputs=log_probs,
            targets=tokens,
            decoder_lengths=encoded_len,
        )
        
        # Store for epoch end
        if isinstance(self.validation_step_outputs, list):
            if dataloader_idx >= len(self.validation_step_outputs):
                self.validation_step_outputs.extend([[]] * (dataloader_idx + 1 - len(self.validation_step_outputs)))
            self.validation_step_outputs[dataloader_idx].append({'val_loss': loss_value})
        else:
            self.validation_step_outputs.append({'val_loss': loss_value})
        
        self.log('val_loss', loss_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'val_loss': loss_value}
    
    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """
        Test step.
        
        Args:
            batch: Batch data (same format as training_step)
            batch_idx: Batch index
            dataloader_idx: Dataloader index
        
        Returns:
            dict: {'test_loss': loss_value}
        """
        # Handle different batch formats
        if isinstance(batch, ssl_dataset.AudioNoiseBatch):
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=batch.audio,
                input_signal_length=batch.audio_len,
                noisy_input_signal=batch.noisy_audio,
                noisy_input_signal_length=batch.noisy_audio_len,
                apply_mask=True,
            )
        elif isinstance(batch, (tuple, list)) and len(batch) >= 6:
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=batch[0],
                input_signal_length=batch[1],
                noisy_input_signal=batch[4],
                noisy_input_signal_length=batch[5],
                apply_mask=True,
            )
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
        
        # Compute loss
        loss_value = self.loss(
            masks=masks,
            decoder_outputs=log_probs,
            targets=tokens,
            decoder_lengths=encoded_len,
        )
        
        # Store for epoch end
        if isinstance(self.test_step_outputs, list):
            if dataloader_idx >= len(self.test_step_outputs):
                self.test_step_outputs.extend([[]] * (dataloader_idx + 1 - len(self.test_step_outputs)))
            self.test_step_outputs[dataloader_idx].append({'test_loss': loss_value})
        else:
            self.test_step_outputs.append({'test_loss': loss_value})
        
        self.log('test_loss', loss_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'test_loss': loss_value}
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if not self.validation_step_outputs:
            return
        
        # Handle single or multiple dataloaders
        if isinstance(self.validation_step_outputs[0], dict):
            # Single dataloader
            val_losses = [x['val_loss'] for x in self.validation_step_outputs]
            val_loss_mean = torch.stack(val_losses).mean()
            self.log('val_loss', val_loss_mean, on_epoch=True, sync_dist=True)
            self.validation_step_outputs.clear()
        else:
            # Multiple dataloaders
            for dataloader_idx, outputs in enumerate(self.validation_step_outputs):
                if outputs:
                    val_losses = [x['val_loss'] for x in outputs]
                    val_loss_mean = torch.stack(val_losses).mean()
                    self.log(f'val_loss_dl{dataloader_idx}', val_loss_mean, on_epoch=True, sync_dist=True)
                    self.validation_step_outputs[dataloader_idx].clear()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        if not self.test_step_outputs:
            return
        
        # Handle single or multiple dataloaders
        if isinstance(self.test_step_outputs[0], dict):
            # Single dataloader
            test_losses = [x['test_loss'] for x in self.test_step_outputs]
            test_loss_mean = torch.stack(test_losses).mean()
            self.log('test_loss', test_loss_mean, on_epoch=True, sync_dist=True)
            self.test_step_outputs.clear()
        else:
            # Multiple dataloaders
            for dataloader_idx, outputs in enumerate(self.test_step_outputs):
                if outputs:
                    test_losses = [x['test_loss'] for x in outputs]
                    test_loss_mean = torch.stack(test_losses).mean()
                    self.log(f'test_loss_dl{dataloader_idx}', test_loss_mean, on_epoch=True, sync_dist=True)
                    self.test_step_outputs[dataloader_idx].clear()
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Users should override this method or configure optimizers via config.
        This is a placeholder that returns None - users must implement their own.
        """
        # This should be implemented by users or via config
        # For compatibility, you can use ModelPT's optimizer setup
        return None

