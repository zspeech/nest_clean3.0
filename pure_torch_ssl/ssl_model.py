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
Pure PyTorch SSL Model - Standalone implementation.

No PyTorch Lightning, no NeMo, no OmegaConf dependencies.
Only requires: torch, numpy, yaml, librosa (for audio preprocessing)

Usage:
    from ssl_model import PureTorchSSLModel
    
    # From config file
    model = PureTorchSSLModel.from_config_file("config.yaml")
    
    # Or with explicit parameters
    model = PureTorchSSLModel(
        sample_rate=16000,
        features=80,
        n_layers=17,
        d_model=512,
        ...
    )
    
    # Training
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for batch in dataloader:
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import numpy as np
import yaml

# Local imports
from modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from modules.conformer_encoder import ConformerEncoder
from modules.ssl_modules.masking import RandomBlockMasking, ConvFeatureMaksingWrapper
from modules.ssl_modules.quantizers import RandomProjectionVectorQuantizer
from modules.ssl_modules.multi_softmax_decoder import MultiSoftmaxDecoder
from losses.ssl_losses.mlm import MultiMLMLoss

__all__ = ['PureTorchSSLModel', 'create_optimizer', 'create_scheduler']


def _load_yaml_config(config_path: str) -> dict:
    """Load YAML config and resolve ${...} interpolations."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    def resolve_interpolations(obj, root):
        """Recursively resolve ${path.to.value} interpolations."""
        if isinstance(obj, dict):
            return {k: resolve_interpolations(v, root) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_interpolations(v, root) for v in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            path = obj[2:-1].split('.')
            val = root
            for p in path:
                if isinstance(val, dict):
                    val = val.get(p, obj)
                else:
                    return obj
            return val
        return obj
    
    return resolve_interpolations(cfg, cfg)


def _get_nested(cfg: dict, *keys, default=None):
    """Safely get nested dict value."""
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val if val is not None else default


class PureTorchSSLModel(nn.Module):
    """
    Pure PyTorch SSL Model for denoising and masked token prediction.
    
    This is the NEST (Noise-resilient Early-fusion Speech Tokenizer) model
    implemented in pure PyTorch without any framework dependencies.
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
        super().__init__()
        
        # Save config for checkpointing
        self._config = {
            'sample_rate': sample_rate, 'features': features, 'window_size': window_size,
            'window_stride': window_stride, 'n_fft': n_fft, 'normalize': normalize,
            'n_layers': n_layers, 'd_model': d_model, 'n_heads': n_heads,
            'subsampling': subsampling, 'subsampling_factor': subsampling_factor,
            'subsampling_conv_channels': subsampling_conv_channels,
            'ff_expansion_factor': ff_expansion_factor, 'conv_kernel_size': conv_kernel_size,
            'dropout': dropout, 'num_classes': num_classes, 'num_books': num_books,
            'code_dim': code_dim, 'squeeze_single': squeeze_single, 'block_size': block_size,
            'mask_prob': mask_prob, 'mask_threshold': mask_threshold, 'mask_position': mask_position,
        }
        
        # Preprocessor
        print("Initializing preprocessor...")
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate,
            normalize=normalize,
            window_size=window_size,
            window_stride=window_stride,
            features=features,
            n_fft=n_fft,
            log=True,
            frame_splicing=1,
            dither=0.0,
            pad_to=16,
            pad_value=0.0,
        )
        
        # Quantizer
        print("Initializing quantizer...")
        self.quantizer = RandomProjectionVectorQuantizer(
            feat_in=features,
            code_dim=code_dim,
            num_books=num_books,
            num_classes=num_classes,
            dist_fn="l2",
            freeze=True,
            squeeze_single=squeeze_single,
            combine_time_steps=subsampling_factor,
        )
        
        # Mask processor
        print("Initializing mask_processor...")
        self.mask_processor = RandomBlockMasking(
            block_size=block_size,
            mask_prob=mask_prob,
            feat_in=features,
            freeze=True,
            allow_overlap=True,
        )
        
        # Encoder
        print("Initializing encoder...")
        self.encoder = ConformerEncoder(
            feat_in=features,
            feat_out=-1,
            n_layers=n_layers,
            d_model=d_model,
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            subsampling_conv_channels=subsampling_conv_channels,
            ff_expansion_factor=ff_expansion_factor,
            self_attention_model="rel_pos",
            n_heads=n_heads,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            dropout_pre_encoder=dropout,
            dropout_emb=0.0,
            dropout_att=dropout,
        )
        
        # Decoder
        print("Initializing decoder...")
        self.decoder = MultiSoftmaxDecoder(
            feat_in=d_model,
            num_classes=num_classes,
            num_decoders=num_books,
            squeeze_single=squeeze_single,
            use_bias=True,
        )
        
        # Loss
        print("Initializing loss...")
        self.loss = MultiMLMLoss(
            combine_time_steps=subsampling_factor,
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
        """Create model from YAML config file."""
        print(f"Loading config from: {config_path}")
        cfg = _load_yaml_config(config_path)
        model_cfg = cfg.get('model', cfg)
        
        return cls(
            sample_rate=model_cfg.get('sample_rate', 16000),
            features=_get_nested(model_cfg, 'preprocessor', 'features', default=80),
            window_size=_get_nested(model_cfg, 'preprocessor', 'window_size', default=0.025),
            window_stride=_get_nested(model_cfg, 'preprocessor', 'window_stride', default=0.01),
            n_fft=_get_nested(model_cfg, 'preprocessor', 'n_fft', default=512),
            normalize=_get_nested(model_cfg, 'preprocessor', 'normalize', default='per_feature'),
            n_layers=_get_nested(model_cfg, 'encoder', 'n_layers', default=17),
            d_model=_get_nested(model_cfg, 'encoder', 'd_model', default=512),
            n_heads=_get_nested(model_cfg, 'encoder', 'n_heads', default=8),
            subsampling=_get_nested(model_cfg, 'encoder', 'subsampling', default='dw_striding'),
            subsampling_factor=_get_nested(model_cfg, 'encoder', 'subsampling_factor', default=8),
            subsampling_conv_channels=_get_nested(model_cfg, 'encoder', 'subsampling_conv_channels', default=256),
            ff_expansion_factor=_get_nested(model_cfg, 'encoder', 'ff_expansion_factor', default=4),
            conv_kernel_size=_get_nested(model_cfg, 'encoder', 'conv_kernel_size', default=9),
            dropout=_get_nested(model_cfg, 'encoder', 'dropout', default=0.1),
            num_classes=model_cfg.get('num_classes', 8192),
            num_books=model_cfg.get('num_books', 1),
            code_dim=model_cfg.get('code_dim', 16),
            squeeze_single=model_cfg.get('squeeze_single', False),
            block_size=_get_nested(model_cfg, 'masking', 'block_size', default=40),
            mask_prob=_get_nested(model_cfg, 'masking', 'mask_prob', default=0.01),
            mask_threshold=_get_nested(model_cfg, 'loss', 'mask_threshold', default=0.8),
            mask_position=model_cfg.get('mask_position', 'pre_conv'),
        )
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> 'PureTorchSSLModel':
        """Create model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def forward(
        self,
        audio: torch.Tensor,
        audio_len: torch.Tensor,
        noisy_audio: torch.Tensor,
        noisy_audio_len: torch.Tensor,
        apply_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            audio: Clean audio waveform [B, T]
            audio_len: Length of each clean audio [B]
            noisy_audio: Noisy audio waveform [B, T]
            noisy_audio_len: Length of each noisy audio [B]
            apply_mask: Whether to apply random masking
        
        Returns:
            Dict containing:
                - loss: Computed MLM loss
                - log_probs: Log probabilities from decoder [B, T, C, H]
                - encoded_len: Encoded sequence lengths [B]
                - masks: Applied masks [B, D, T]
                - tokens: Quantized target tokens [B, T, H]
        
        Usage:
            # Training
            outputs = model(audio, audio_len, noisy_audio, noisy_audio_len)
            outputs['loss'].backward()
            optimizer.step()
            
            # Validation (no grad)
            with torch.no_grad():
                outputs = model(audio, audio_len, noisy_audio, noisy_audio_len)
                val_loss = outputs['loss'].item()
        """
        # 1. Preprocess clean audio -> mel spectrogram for targets
        processed_clean, _ = self.preprocessor(input_signal=audio, length=audio_len)
        
        # 2. Get quantized tokens from clean audio
        _, tokens = self.quantizer(input_signal=processed_clean)
        
        # 3. Preprocess noisy audio -> mel spectrogram for encoder input
        processed_noisy, processed_noisy_len = self.preprocessor(
            input_signal=noisy_audio, length=noisy_audio_len
        )
        
        # 4. Apply masking and encode
        if self.pre_encoder is not None:
            # Post-conv masking mode
            self.pre_encoder.set_masking_enabled(apply_mask=apply_mask)
            encoded, encoded_len = self.encoder(
                audio_signal=processed_noisy, length=processed_noisy_len
            )
            masks = self.pre_encoder.get_current_mask()
        else:
            # Pre-conv masking mode (default)
            if apply_mask:
                masked_signal, masks = self.mask_processor(
                    input_feats=processed_noisy, input_lengths=processed_noisy_len
                )
            else:
                masked_signal = processed_noisy
                masks = torch.zeros_like(processed_noisy)
            
            encoded, encoded_len = self.encoder(
                audio_signal=masked_signal, length=processed_noisy_len
            )
        
        # 5. Decode
        log_probs = self.decoder(encoder_output=encoded)
        
        # 6. Compute loss
        loss = self.loss(
            masks=masks,
            decoder_outputs=log_probs,
            targets=tokens,
            decoder_lengths=encoded_len,
        )
        
        return {
            'loss': loss,
            'log_probs': log_probs,
            'encoded_len': encoded_len,
            'masks': masks,
            'tokens': tokens,
        }
    
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


def create_optimizer(model: nn.Module, cfg: dict, lr: float = None) -> torch.optim.Optimizer:
    """Create optimizer from config dict."""
    model_cfg = cfg.get('model', cfg)
    optim_cfg = model_cfg.get('optim', {})
    optim_name = str(optim_cfg.get('name', 'adamw')).lower()
    if lr is None:
        lr = float(optim_cfg.get('lr', 1e-4))
    betas = optim_cfg.get('betas', [0.9, 0.999])
    weight_decay = float(optim_cfg.get('weight_decay', 0.0))
    
    if isinstance(betas, list):
        betas = tuple(float(b) for b in betas)
    
    if optim_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optim_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optim_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(optim_cfg.get('momentum', 0.9)), weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, num_training_steps: int = None):
    """Create learning rate scheduler from config dict."""
    model_cfg = cfg.get('model', cfg)
    optim_cfg = model_cfg.get('optim', {})
    sched_cfg = optim_cfg.get('sched', None)
    
    if sched_cfg is None:
        return None
    
    sched_name = str(sched_cfg.get('name', 'noam')).lower()
    
    if sched_name in ('noamannealing', 'noam'):
        d_model = int(sched_cfg.get('d_model', _get_nested(model_cfg, 'encoder', 'd_model', default=512)))
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
        max_steps = num_training_steps or 100000
        
        def cosine_with_warmup(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)
    
    return None

