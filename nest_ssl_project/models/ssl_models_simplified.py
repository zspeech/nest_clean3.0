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
that maintains full compatibility and alignment with the original model.

Key features:
- Inherits from ModelPT for optimizer/scheduler support
- Same forward pass logic as original (bit-exact output)
- Simplified data loader setup
- Removes unnecessary mixins while keeping core functionality
"""

from typing import Optional, Dict, Any, Union, List
from math import ceil
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf

# Import from local modules
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data import ssl_dataset
from data import audio_to_text_dataset
from modules.ssl_modules.masking import ConvFeatureMaksingWrapper
from core.classes import ModelPT
from core.classes.common import typecheck
from core.neural_types import (
    AudioSignal,
    LabelsType,
    LengthsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)
from utils.logging import get_logger, is_global_rank_zero

__all__ = ['SimplifiedSSLModel']

logging = get_logger(__name__)


class SimplifiedSSLModel(ModelPT):
    """
    Simplified SSL Model for denoising and masked token prediction.
    
    This is a streamlined version of EncDecDenoiseMaskedTokenPredModel that:
    - Maintains bit-exact forward pass alignment with original
    - Supports optimizer/scheduler configuration via ModelPT
    - Simplifies data loader setup
    - Removes AccessMixin and ASRModuleMixin complexity
    
    Usage:
        model = SimplifiedSSLModel(cfg=cfg.model, trainer=trainer)
        trainer.fit(model)
    """
    
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initialize simplified SSL model.
        
        Args:
            cfg: Model configuration (same format as EncDecDenoiseMaskedTokenPredModel)
            trainer: PyTorch Lightning trainer
        """
        # Get world size for data loading
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size
        
        # Initialize ModelPT (handles optimizer setup, etc.)
        super().__init__(cfg=cfg, trainer=trainer)
        
        # Initialize preprocessor and encoder (same order as original)
        self.preprocessor = self.from_config_dict(self._cfg.preprocessor)
        
        # Handle mask_position config adjustment BEFORE creating components
        if self._cfg.get("mask_position", "pre_conv") == "post_conv":
            # Adjust config for post-convolution masking
            self._cfg.quantizer.feat_in = self._cfg.encoder.d_model
            self._cfg.masking.feat_in = self._cfg.encoder.d_model
            self._cfg.masking.block_size = self._cfg.masking.block_size // self._cfg.encoder.subsampling_factor
            self._cfg.loss.combine_time_steps = 1
        
        # Initialize components (same order as EncDecMaskedTokenPredModel)
        self.quantizer = self.from_config_dict(self._cfg.quantizer)
        self.mask_processor = self.from_config_dict(self._cfg.masking)
        self.encoder = self.from_config_dict(self._cfg.encoder)
        self.decoder = self.from_config_dict(self._cfg.decoder)
        self.loss = self.from_config_dict(self._cfg.loss)
        
        # Handle post-conv masking wrapper
        self.pre_encoder = None
        if self._cfg.get("mask_position", "pre_conv") == "post_conv":
            self.pre_encoder = ConvFeatureMaksingWrapper(self.encoder.pre_encode, self.mask_processor)
            self.encoder.pre_encode = self.pre_encoder
        
        # Initialize validation/test outputs lists
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        """Set up dataloader from config (simplified version)."""
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        
        dataset = ssl_dataset.get_audio_noise_dataset_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )
        
        if dataset is None:
            return None
        
        shuffle = config.get('shuffle', True)
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False
        
        # Get collate_fn
        collate_fn = None
        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset, 'datasets') and len(dataset.datasets) > 0:
            if hasattr(dataset.datasets[0], 'collate_fn'):
                collate_fn = dataset.datasets[0].collate_fn
        
        num_workers = config.get('num_workers', 0)
        pin_memory = config.get('pin_memory', False)
        if num_workers == 0:
            pin_memory = False
        
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """Set up training data loader."""
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)
        
        # Handle IterableDataset batch count
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
    
    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """Set up validation data loader."""
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False
        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)
    
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "noisy_input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "noisy_input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_noisy_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_noisy_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "apply_mask": NeuralType(optional=True),
        }
    
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        if self._cfg.num_books == 1 and self._cfg.squeeze_single:
            logprobs = NeuralType(('B', 'T', 'C'), LogprobsType())
            tokens = NeuralType(('B', 'T'), LabelsType())
        else:
            logprobs = NeuralType(('B', 'T', 'C', 'H'), LogprobsType())
            tokens = NeuralType(('B', 'T', 'H'), LabelsType())
        return {
            "logprobs": logprobs,
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
            "masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "tokens": tokens,
        }
    
    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        noise_signal=None,  # noqa - kept for API compatibility
        noise_signal_length=None,  # noqa
        processed_noise_signal=None,  # noqa
        processed_noise_signal_length=None,  # noqa
        noisy_input_signal=None,
        noisy_input_signal_length=None,
        processed_noisy_input_signal=None,
        processed_noisy_input_signal_length=None,
        apply_mask=False,
    ):
        """
        Forward pass - EXACTLY matches EncDecDenoiseMaskedTokenPredModel.forward()
        
        Args:
            input_signal: Clean audio [B, T]
            input_signal_length: Clean audio lengths [B]
            processed_signal: Preprocessed clean features [B, D, T]
            processed_signal_length: Preprocessed clean lengths [B]
            noise_signal: Noise audio (unused, for API compatibility)
            noise_signal_length: Noise lengths (unused)
            processed_noise_signal: Preprocessed noise (unused)
            processed_noise_signal_length: Preprocessed noise lengths (unused)
            noisy_input_signal: Noisy audio [B, T]
            noisy_input_signal_length: Noisy audio lengths [B]
            processed_noisy_input_signal: Preprocessed noisy features [B, D, T]
            processed_noisy_input_signal_length: Preprocessed noisy lengths [B]
            apply_mask: Whether to apply masking
        
        Returns:
            tuple: (log_probs, encoded_len, masks, tokens)
        """
        # Process clean signal
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )
        
        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        
        # Process noisy signal
        has_noisy_input_signal = noisy_input_signal is not None and noisy_input_signal_length is not None
        has_processed_noisy_input_signal = (
            processed_noisy_input_signal is not None and processed_noisy_input_signal_length is not None
        )
        if (has_noisy_input_signal ^ has_processed_noisy_input_signal) == False:
            raise ValueError(
                f"{self} Arguments ``noisy_input_signal`` and ``noisy_input_signal_length`` are mutually exclusive "
                " with ``processed_noisy_input_signal`` and ``processed_noisy_input_signal_len`` arguments."
            )
        if not has_processed_noisy_input_signal:
            processed_noisy_input_signal, processed_noisy_input_signal_length = self.preprocessor(
                input_signal=noisy_input_signal,
                length=noisy_input_signal_length,
            )
        
        # Core forward logic - EXACTLY matches original
        if self.pre_encoder is not None:
            # mask after convolutional sub-sampling
            feats, _ = self.pre_encoder.pre_encode(x=processed_signal, lengths=processed_signal_length)
            _, tokens = self.quantizer(input_signal=feats.transpose(1, 2))
            self.pre_encoder.set_masking_enabled(apply_mask=apply_mask)
            
            encoded, encoded_len = self.encoder(
                audio_signal=processed_noisy_input_signal, length=processed_noisy_input_signal_length
            )
            masks = self.pre_encoder.get_current_mask()
        else:
            _, tokens = self.quantizer(input_signal=processed_signal)
            if apply_mask:
                masked_signal, masks = self.mask_processor(
                    input_feats=processed_noisy_input_signal, input_lengths=processed_noisy_input_signal_length
                )
            else:
                masked_signal = processed_noisy_input_signal
                masks = torch.zeros_like(processed_noisy_input_signal)
            
            encoded, encoded_len = self.encoder(audio_signal=masked_signal, length=processed_noisy_input_signal_length)
        
        log_probs = self.decoder(encoder_output=encoded)
        
        return log_probs, encoded_len, masks, tokens
    
    def training_step(self, batch: ssl_dataset.AudioNoiseBatch, batch_idx: int):
        """Training step - matches original exactly."""
        log_probs, encoded_len, masks, tokens = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noise_signal=batch.noise,
            noise_signal_length=batch.noise_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=True,
        )
        
        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)
        
        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'] if self._optimizer is not None else 0.0,
            'global_step': self.trainer.global_step if self.trainer is not None else 0,
            'train_loss': loss_value,
        }
        
        return {'loss': loss_value, 'log': tensorboard_logs}
    
    @torch.no_grad()
    def inference_pass(
        self,
        batch: ssl_dataset.AudioNoiseBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
        mode: str = 'val',
        apply_mask: bool = True,
    ):
        """Inference pass for validation/test."""
        log_probs, encoded_len, masks, tokens = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noise_signal=batch.noise,
            noise_signal_length=batch.noise_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=apply_mask,
        )
        
        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)
        
        return {f'{mode}_loss': loss_value}
    
    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        """Validation step."""
        metrics = self.inference_pass(batch, batch_idx, dataloader_idx, apply_mask=True)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            if len(self.validation_step_outputs) <= dataloader_idx:
                self.validation_step_outputs.extend([[]] * (dataloader_idx + 1 - len(self.validation_step_outputs)))
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx=0, dataloader_idx=0):
        """Test step."""
        metrics = self.inference_pass(batch, batch_idx, dataloader_idx, mode="test", apply_mask=True)
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            if len(self.test_step_outputs) <= dataloader_idx:
                self.test_step_outputs.extend([[]] * (dataloader_idx + 1 - len(self.test_step_outputs)))
            self.test_step_outputs[dataloader_idx].append(metrics)
        else:
            self.test_step_outputs.append(metrics)
        return metrics
    
    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        """Aggregate validation outputs."""
        loss_list = []
        for i, x in enumerate(outputs):
            if not isinstance(x, dict):
                logging.warning(f'Batch {i} output is not a dictionary: {x}')
            if 'val_loss' in x:
                loss_list.append(x['val_loss'])
        
        if len(loss_list) == 0:
            return {}
        
        val_loss_mean = torch.stack(loss_list).mean()
        tensorboard_logs = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}
    
    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        """Aggregate test outputs."""
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': test_loss_mean}
        return {'test_loss': test_loss_mean, 'log': tensorboard_logs}
    
    def on_validation_epoch_end(self):
        """Called at end of validation epoch."""
        if not self.validation_step_outputs:
            return {}
        
        if isinstance(self.validation_step_outputs[0], dict):
            output_dict = self.multi_validation_epoch_end(self.validation_step_outputs, dataloader_idx=0)
            if output_dict and 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)
            self.validation_step_outputs.clear()
            return output_dict
        else:
            for dataloader_idx, val_outputs in enumerate(self.validation_step_outputs):
                if len(val_outputs) > 0:
                    self.multi_validation_epoch_end(val_outputs, dataloader_idx=dataloader_idx)
                    self.validation_step_outputs[dataloader_idx].clear()
        return {}
    
    def on_test_epoch_end(self):
        """Called at end of test epoch."""
        if not self.test_step_outputs:
            return {}
        
        if isinstance(self.test_step_outputs[0], dict):
            output_dict = self.multi_test_epoch_end(self.test_step_outputs, dataloader_idx=0)
            if output_dict and 'test_loss' in output_dict:
                self.log('test_loss', output_dict['test_loss'], on_epoch=True, sync_dist=True)
            self.test_step_outputs.clear()
            return output_dict
        else:
            for dataloader_idx, test_outputs in enumerate(self.test_step_outputs):
                if len(test_outputs) > 0:
                    self.multi_test_epoch_end(test_outputs, dataloader_idx=dataloader_idx)
                    self.test_step_outputs[dataloader_idx].clear()
        return {}
    
    @classmethod
    def list_available_models(cls):
        """List available models (none for simplified version)."""
        return []
    
    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """Transfer batch to device with non_blocking for speed."""
        from utils.device_utils import move_data_to_device
        return move_data_to_device(batch, device, non_blocking=True)
