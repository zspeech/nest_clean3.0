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
Pure PyTorch SSL Model - No Lightning, no Serialization dependency.

This is a standalone implementation that:
- Directly imports all modules
- Reads config from YAML file
- Single GPU training
- Simple training loop interface
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Direct module imports (no Serialization)
from modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from modules.conformer_encoder import ConformerEncoder
from modules.ssl_modules.masking import RandomBlockMasking, ConvFeatureMaksingWrapper
from modules.ssl_modules.quantizers import RandomProjectionVectorQuantizer
from modules.ssl_modules.multi_softmax_decoder import MultiSoftmaxDecoder
from losses.ssl_losses.mlm import MultiMLMLoss

__all__ = ['PureTorchSSLModel']


class PureTorchSSLModel(nn.Module):
    """
    Pure PyTorch SSL Model for denoising and masked token prediction.
    
    No PyTorch Lightning dependency. No Serialization dependency.
    Directly imports and instantiates all modules.
    
    Usage:
        # From config file
        model = PureTorchSSLModel.from_config_file("config/nest_fast-conformer.yaml")
        model.to('cuda')
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        for batch in dataloader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """
    
    def __init__(
        self,
        # Preprocessor params
        sample_rate: int = 16000,
        features: int = 80,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        n_fft: int = 512,
        normalize: str = "per_feature",
        # Encoder params
        n_layers: int = 17,
        d_model: int = 512,
        n_heads: int = 8,
        subsampling: str = "dw_striding",
        subsampling_factor: int = 8,
        subsampling_conv_channels: int = 256,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 9,
        dropout: float = 0.1,
        # Quantizer params
        num_classes: int = 8192,
        num_books: int = 1,
        code_dim: int = 16,
        squeeze_single: bool = False,
        # Masking params
        block_size: int = 40,
        mask_prob: float = 0.01,
        # Loss params
        mask_threshold: float = 0.8,
        # Mask position
        mask_position: str = "pre_conv",
    ):
        """
        Initialize pure PyTorch SSL model with explicit parameters.
        
        Args:
            sample_rate: Audio sample rate
            features: Number of mel features
            window_size: FFT window size in seconds
            window_stride: FFT window stride in seconds
            n_fft: FFT size
            normalize: Normalization type
            n_layers: Number of conformer layers
            d_model: Model dimension
            n_heads: Number of attention heads
            subsampling: Subsampling type
            subsampling_factor: Subsampling factor
            subsampling_conv_channels: Subsampling conv channels
            ff_expansion_factor: Feed forward expansion factor
            conv_kernel_size: Convolution kernel size
            dropout: Dropout rate
            num_classes: Number of quantizer classes
            num_books: Number of codebooks
            code_dim: Code dimension
            squeeze_single: Whether to squeeze single codebook
            block_size: Masking block size
            mask_prob: Masking probability
            mask_threshold: Loss mask threshold
            mask_position: "pre_conv" or "post_conv"
        """
        super().__init__()
        
        self.mask_position = mask_position
        self.subsampling_factor = subsampling_factor
        
        # Store config for saving
        self._config = {
            'sample_rate': sample_rate,
            'features': features,
            'window_size': window_size,
            'window_stride': window_stride,
            'n_fft': n_fft,
            'normalize': normalize,
            'n_layers': n_layers,
            'd_model': d_model,
            'n_heads': n_heads,
            'subsampling': subsampling,
            'subsampling_factor': subsampling_factor,
            'subsampling_conv_channels': subsampling_conv_channels,
            'ff_expansion_factor': ff_expansion_factor,
            'conv_kernel_size': conv_kernel_size,
            'dropout': dropout,
            'num_classes': num_classes,
            'num_books': num_books,
            'code_dim': code_dim,
            'squeeze_single': squeeze_single,
            'block_size': block_size,
            'mask_prob': mask_prob,
            'mask_threshold': mask_threshold,
            'mask_position': mask_position,
        }
        
        # Adjust for post_conv masking
        masking_feat_in = features
        masking_block_size = block_size
        loss_combine_time_steps = subsampling_factor
        
        if mask_position == "post_conv":
            masking_feat_in = d_model
            masking_block_size = block_size // subsampling_factor
            loss_combine_time_steps = 1
        
        # Initialize preprocessor
        print("Initializing preprocessor...")
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate,
            features=features,
            window_size=window_size,
            window_stride=window_stride,
            n_fft=n_fft,
            normalize=normalize,
            window="hann",
            log=True,
            frame_splicing=1,
            dither=0.0,
            pad_to=16,
            pad_value=0.0,
        )
        
        # Initialize quantizer
        print("Initializing quantizer...")
        self.quantizer = RandomProjectionVectorQuantizer(
            feat_in=features if mask_position == "pre_conv" else d_model,
            code_dim=code_dim,
            num_books=num_books,
            num_classes=num_classes,
            dist_fn="l2",
            freeze=True,
            squeeze_single=squeeze_single,
            combine_time_steps=subsampling_factor if mask_position == "pre_conv" else 1,
        )
        
        # Initialize mask processor
        print("Initializing mask_processor...")
        self.mask_processor = RandomBlockMasking(
            block_size=masking_block_size,
            mask_prob=mask_prob,
            feat_in=masking_feat_in,
            freeze=True,
            allow_overlap=True,
        )
        
        # Initialize encoder
        print("Initializing encoder...")
        self.encoder = ConformerEncoder(
            feat_in=features,
            feat_out=-1,
            n_layers=n_layers,
            d_model=d_model,
            use_bias=True,
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            subsampling_conv_channels=subsampling_conv_channels,
            causal_downsampling=False,
            ff_expansion_factor=ff_expansion_factor,
            self_attention_model="rel_pos",
            n_heads=n_heads,
            att_context_size=[-1, -1],
            att_context_style="regular",
            xscaling=True,
            untie_biases=True,
            pos_emb_max_len=5000,
            conv_kernel_size=conv_kernel_size,
            conv_norm_type="batch_norm",
            conv_context_size=None,
            dropout=dropout,
            dropout_pre_encoder=dropout,
            dropout_emb=0.0,
            dropout_att=dropout,
            stochastic_depth_drop_prob=0.0,
            stochastic_depth_mode="linear",
            stochastic_depth_start_layer=1,
        )
        
        # Initialize decoder
        print("Initializing decoder...")
        self.decoder = MultiSoftmaxDecoder(
            feat_in=d_model,
            num_classes=num_classes,
            num_decoders=num_books,
            squeeze_single=squeeze_single,
            use_bias=True,
        )
        
        # Initialize loss
        print("Initializing loss...")
        self.loss = MultiMLMLoss(
            combine_time_steps=loss_combine_time_steps,
            mask_threshold=mask_threshold,
            num_decoders=num_books,
            squeeze_single=squeeze_single,
        )
        
        # Handle post-conv masking wrapper
        self.pre_encoder = None
        if mask_position == "post_conv":
            print("Setting up post-conv masking wrapper...")
            self.pre_encoder = ConvFeatureMaksingWrapper(self.encoder.pre_encode, self.mask_processor)
            self.encoder.pre_encode = self.pre_encoder
        
        print(f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'PureTorchSSLModel':
        """
        Create model from YAML config file.
        
        Args:
            config_path: Path to YAML config file
        
        Returns:
            PureTorchSSLModel instance
        """
        print(f"Loading config from: {config_path}")
        cfg = OmegaConf.load(config_path)
        
        # Extract model config
        model_cfg = cfg.model if 'model' in cfg else cfg
        
        # Resolve interpolations
        OmegaConf.resolve(cfg)
        model_cfg = cfg.model if 'model' in cfg else cfg
        
        return cls(
            # Preprocessor
            sample_rate=model_cfg.get('sample_rate', 16000),
            features=model_cfg.preprocessor.get('features', 80),
            window_size=model_cfg.preprocessor.get('window_size', 0.025),
            window_stride=model_cfg.preprocessor.get('window_stride', 0.01),
            n_fft=model_cfg.preprocessor.get('n_fft', 512),
            normalize=model_cfg.preprocessor.get('normalize', 'per_feature'),
            # Encoder
            n_layers=model_cfg.encoder.get('n_layers', 17),
            d_model=model_cfg.encoder.get('d_model', 512),
            n_heads=model_cfg.encoder.get('n_heads', 8),
            subsampling=model_cfg.encoder.get('subsampling', 'dw_striding'),
            subsampling_factor=model_cfg.encoder.get('subsampling_factor', 8),
            subsampling_conv_channels=model_cfg.encoder.get('subsampling_conv_channels', 256),
            ff_expansion_factor=model_cfg.encoder.get('ff_expansion_factor', 4),
            conv_kernel_size=model_cfg.encoder.get('conv_kernel_size', 9),
            dropout=model_cfg.encoder.get('dropout', 0.1),
            # Quantizer
            num_classes=model_cfg.get('num_classes', 8192),
            num_books=model_cfg.get('num_books', 1),
            code_dim=model_cfg.get('code_dim', 16),
            squeeze_single=model_cfg.get('squeeze_single', False),
            # Masking
            block_size=model_cfg.masking.get('block_size', 40),
            mask_prob=model_cfg.masking.get('mask_prob', 0.01),
            # Loss
            mask_threshold=model_cfg.loss.get('mask_threshold', 0.8),
            # Mask position
            mask_position=model_cfg.get('mask_position', 'pre_conv'),
        )
    
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
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
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
        """Compute loss from forward outputs."""
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
                - Object with .audio, .audio_len, .noisy_audio, .noisy_audio_len
                - Tuple: (audio, audio_len, noise, noise_len, noisy_audio, noisy_audio_len)
                - Dict with keys: 'audio', 'audio_len', 'noisy_audio', 'noisy_audio_len'
        
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
        return self.compute_loss(log_probs, encoded_len, masks, tokens)
    
    @torch.no_grad()
    def validation_step(self, batch) -> torch.Tensor:
        """Single validation step (no gradients)."""
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
        
        log_probs, encoded_len, masks, tokens = self.forward(
            input_signal=audio,
            input_signal_length=audio_len,
            noisy_input_signal=noisy_audio,
            noisy_input_signal_length=noisy_audio_len,
            apply_mask=True,
        )
        
        return self.compute_loss(log_probs, encoded_len, masks, tokens)
    
    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, step: int = 0):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self._config,
            'epoch': epoch,
            'step': step,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str, optimizer=None, strict: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"Loaded model state from {path}")
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
        }


def create_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """
    Create optimizer from config.
    
    Args:
        model: The model to optimize
        cfg: Config containing optim section with name, lr, betas, weight_decay
    
    Returns:
        Optimizer instance
    """
    optim_cfg = cfg.model.optim
    optim_name = optim_cfg.get('name', 'adamw').lower()
    lr = optim_cfg.get('lr', 1e-4)
    betas = optim_cfg.get('betas', [0.9, 0.999])
    weight_decay = optim_cfg.get('weight_decay', 0.0)
    
    # Convert OmegaConf list to tuple
    if hasattr(betas, '__iter__') and not isinstance(betas, tuple):
        betas = tuple(betas)
    
    if optim_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optim_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optim_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optim_cfg.get('momentum', 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig, num_training_steps: int = None):
    """
    Create learning rate scheduler from config.
    
    Args:
        optimizer: The optimizer to schedule
        cfg: Config containing optim.sched section
        num_training_steps: Total training steps (optional)
    
    Returns:
        Scheduler instance or None
    """
    if 'sched' not in cfg.model.optim:
        return None
    
    sched_cfg = cfg.model.optim.sched
    sched_name = sched_cfg.get('name', 'noam').lower()
    
    if sched_name == 'noamannealing' or sched_name == 'noam':
        # Noam/Transformer scheduler with warmup
        d_model = sched_cfg.get('d_model', cfg.model.encoder.d_model)
        warmup_steps = sched_cfg.get('warmup_steps', 10000)
        min_lr = sched_cfg.get('min_lr', 1e-6)
        
        def noam_lambda(step):
            step = max(step, 1)
            scale = d_model ** (-0.5)
            lr_scale = min(step ** (-0.5), step * warmup_steps ** (-1.5))
            return max(scale * lr_scale, min_lr / optimizer.defaults['lr'])
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, noam_lambda)
    
    elif sched_name == 'cosine':
        warmup_steps = sched_cfg.get('warmup_steps', 1000)
        
        def cosine_with_warmup(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)
    
    elif sched_name == 'constant':
        return None  # No scheduling
    
    else:
        print(f"Warning: Unknown scheduler '{sched_name}', using constant LR")
        return None
