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

Usage:
    from ssl_model import PureTorchSSLModel, setup_hparams
    
    # From config
    cfg = setup_hparams(your_config_dict, {})
    model = PureTorchSSLModel(cfg)
    
    # Or from YAML file
    model = PureTorchSSLModel.from_config_file("config.yaml")
    
    # Training
    model.to('cuda')
    outputs = model(audio, audio_len, noisy_audio, noisy_audio_len)
    outputs['loss'].backward()
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

__all__ = ['PureTorchSSLModel', 'Hyperparams', 'setup_hparams', 'create_optimizer', 'create_scheduler']


# ============================================================================
# Hyperparams Configuration System
# ============================================================================

class Hyperparams(dict):
    """Dict subclass that allows attribute-style access (cfg.key)."""
    
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'Hyperparams' object has no attribute '{attr}'")
    
    def __setattr__(self, attr, value):
        self[attr] = value
    
    def __repr__(self):
        return f"Hyperparams({dict.__repr__(self)})"


def setup_hparams(config: dict, overrides: dict = None) -> Hyperparams:
    """
    Convert a nested dict config into Hyperparams with attribute access.
    
    Args:
        config: Dict config (can be nested)
        overrides: Optional dict of overrides
    
    Returns:
        Hyperparams object with cfg.key.subkey access
    """
    H = Hyperparams()
    
    for k, v in config.items():
        if isinstance(v, dict):
            H[k] = setup_hparams(v, {})
        else:
            H[k] = v
    
    if overrides:
        H.update(overrides)
    
    return H


def _load_yaml_config(config_path: str) -> Hyperparams:
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
    
    resolved = resolve_interpolations(cfg, cfg)
    return setup_hparams(resolved, {})


# ============================================================================
# Default Configuration
# ============================================================================

def get_default_config() -> Hyperparams:
    """Get default SSL model configuration."""
    return setup_hparams({
        'sample_rate': 16000,
        'num_classes': 8192,
        'num_books': 1,
        'code_dim': 16,
        'squeeze_single': False,
        'mask_position': 'pre_conv',
        
        'preprocessor': {
            'features': 80,
            'window_size': 0.025,
            'window_stride': 0.01,
            'n_fft': 512,
            'normalize': 'per_feature',
            'log': True,
            'dither': 0.0,
            'pad_to': 16,
        },
        
        'encoder': {
            'n_layers': 17,
            'd_model': 512,
            'n_heads': 8,
            'subsampling': 'dw_striding',
            'subsampling_factor': 8,
            'subsampling_conv_channels': 256,
            'ff_expansion_factor': 4,
            'conv_kernel_size': 9,
            'dropout': 0.1,
            'dropout_pre_encoder': 0.1,
            'dropout_emb': 0.0,
            'dropout_att': 0.1,
        },
        
        'masking': {
            'block_size': 40,
            'mask_prob': 0.01,
            'freeze': True,
            'allow_overlap': True,
        },
        
        'decoder': {
            'use_bias': True,
        },
        
        'loss': {
            'mask_threshold': 0.8,
        },
        
        'optim': {
            'name': 'adamw',
            'lr': 1e-4,
            'betas': [0.9, 0.999],
            'weight_decay': 0.0,
        },
    }, {})


# ============================================================================
# SSL Model
# ============================================================================

class PureTorchSSLModel(nn.Module):
    """
    Pure PyTorch SSL Model for denoising and masked token prediction.
    
    This is the NEST (Noise-resilient Early-fusion Speech Tokenizer) model
    implemented in pure PyTorch without any framework dependencies.
    """
    
    def __init__(self, cfg: Hyperparams):
        """
        Initialize SSL model from config.
        
        Args:
            cfg: Hyperparams config with structure:
                cfg.sample_rate
                cfg.preprocessor.features
                cfg.encoder.d_model
                cfg.masking.block_size
                ...
        """
        super().__init__()
        self.cfg = cfg
        
        # Merge with defaults
        defaults = get_default_config()
        for key in defaults:
            if key not in cfg:
                cfg[key] = defaults[key]
            elif isinstance(defaults[key], Hyperparams) and key in cfg:
                for subkey in defaults[key]:
                    if subkey not in cfg[key]:
                        cfg[key][subkey] = defaults[key][subkey]
        
        # Preprocessor
        print("Initializing preprocessor...")
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=cfg.sample_rate,
            normalize=cfg.preprocessor.normalize,
            window_size=cfg.preprocessor.window_size,
            window_stride=cfg.preprocessor.window_stride,
            features=cfg.preprocessor.features,
            n_fft=cfg.preprocessor.n_fft,
            log=cfg.preprocessor.get('log', True),
            frame_splicing=1,
            dither=cfg.preprocessor.get('dither', 0.0),
            pad_to=cfg.preprocessor.get('pad_to', 16),
            pad_value=0.0,
        )
        
        # Quantizer
        print("Initializing quantizer...")
        self.quantizer = RandomProjectionVectorQuantizer(
            feat_in=cfg.preprocessor.features,
            code_dim=cfg.code_dim,
            num_books=cfg.num_books,
            num_classes=cfg.num_classes,
            dist_fn="l2",
            freeze=True,
            squeeze_single=cfg.squeeze_single,
            combine_time_steps=cfg.encoder.subsampling_factor,
        )
        
        # Mask processor
        print("Initializing mask_processor...")
        self.mask_processor = RandomBlockMasking(
            block_size=cfg.masking.block_size,
            mask_prob=cfg.masking.mask_prob,
            feat_in=cfg.preprocessor.features,
            freeze=cfg.masking.get('freeze', True),
            allow_overlap=cfg.masking.get('allow_overlap', True),
        )
        
        # Encoder
        print("Initializing encoder...")
        self.encoder = ConformerEncoder(
            feat_in=cfg.preprocessor.features,
            feat_out=-1,
            n_layers=cfg.encoder.n_layers,
            d_model=cfg.encoder.d_model,
            subsampling=cfg.encoder.subsampling,
            subsampling_factor=cfg.encoder.subsampling_factor,
            subsampling_conv_channels=cfg.encoder.subsampling_conv_channels,
            ff_expansion_factor=cfg.encoder.ff_expansion_factor,
            self_attention_model="rel_pos",
            n_heads=cfg.encoder.n_heads,
            conv_kernel_size=cfg.encoder.conv_kernel_size,
            dropout=cfg.encoder.dropout,
            dropout_pre_encoder=cfg.encoder.get('dropout_pre_encoder', cfg.encoder.dropout),
            dropout_emb=cfg.encoder.get('dropout_emb', 0.0),
            dropout_att=cfg.encoder.get('dropout_att', cfg.encoder.dropout),
        )
        
        # Decoder
        print("Initializing decoder...")
        self.decoder = MultiSoftmaxDecoder(
            feat_in=cfg.encoder.d_model,
            num_classes=cfg.num_classes,
            num_decoders=cfg.num_books,
            squeeze_single=cfg.squeeze_single,
            use_bias=cfg.decoder.get('use_bias', True),
        )
        
        # Loss
        print("Initializing loss...")
        self.loss = MultiMLMLoss(
            combine_time_steps=cfg.encoder.subsampling_factor,
            mask_threshold=cfg.loss.mask_threshold,
            num_decoders=cfg.num_books,
            squeeze_single=cfg.squeeze_single,
        )
        
        # Handle post-conv masking wrapper
        self.pre_encoder = None
        if cfg.get('mask_position', 'pre_conv') == "post_conv":
            print("Setting up post-conv masking wrapper...")
            self.pre_encoder = ConvFeatureMaksingWrapper(self.encoder.pre_encode, self.mask_processor)
            self.encoder.pre_encode = self.pre_encoder
        
        print(f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'PureTorchSSLModel':
        """Create model from YAML config file."""
        print(f"Loading config from: {config_path}")
        cfg = _load_yaml_config(config_path)
        
        # If config has 'model' key, use it
        if 'model' in cfg:
            cfg = cfg.model
        
        return cls(cfg)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> 'PureTorchSSLModel':
        """Create model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg = setup_hparams(checkpoint['config'], {})
        model = cls(cfg)
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
        # Convert Hyperparams back to dict for saving
        def to_dict(obj):
            if isinstance(obj, Hyperparams):
                return {k: to_dict(v) for k, v in obj.items()}
            return obj
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': to_dict(self.cfg),
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


# ============================================================================
# Optimizer & Scheduler
# ============================================================================

def create_optimizer(model: nn.Module, cfg: Hyperparams) -> torch.optim.Optimizer:
    """
    Create optimizer from config.
    
    Args:
        model: Model to optimize
        cfg: Config with cfg.optim.name, cfg.optim.lr, etc.
    """
    optim_cfg = cfg.get('optim', Hyperparams())
    optim_name = str(optim_cfg.get('name', 'adamw')).lower()
    lr = float(optim_cfg.get('lr', 1e-4))
    weight_decay = float(optim_cfg.get('weight_decay', 0.0))
    betas = optim_cfg.get('betas', [0.9, 0.999])
    
    if isinstance(betas, list):
        betas = tuple(float(b) for b in betas)
    
    if optim_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optim_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif optim_name == 'sgd':
        momentum = float(optim_cfg.get('momentum', 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: Hyperparams, num_training_steps: int = None):
    """
    Create learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer instance
        cfg: Config with cfg.optim.sched.name, etc.
    """
    optim_cfg = cfg.get('optim', Hyperparams())
    sched_cfg = optim_cfg.get('sched', None)
    
    if sched_cfg is None:
        return None
    
    sched_name = str(sched_cfg.get('name', 'noam')).lower()
    
    if sched_name in ('noamannealing', 'noam'):
        d_model = int(sched_cfg.get('d_model', cfg.encoder.d_model))
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
