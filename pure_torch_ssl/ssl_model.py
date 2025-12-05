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
import math
import yaml

# Local imports
from modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from modules.conformer_encoder import ConformerEncoder
from modules.ssl_modules.masking import RandomBlockMasking, ConvFeatureMaksingWrapper
from modules.ssl_modules.quantizers import RandomProjectionVectorQuantizer
from modules.ssl_modules.multi_softmax_decoder import MultiSoftmaxDecoder
from losses.ssl_losses.mlm import MultiMLMLoss
from hparams import Hyperparams, setup_hparams, HPARAMS_REGISTRY

__all__ = ['PureTorchSSLModel', 'create_optimizer', 'create_scheduler', 'get_lr_scheduler']


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
        
        # Merge with defaults from registry if needed
        if 'defaults' in HPARAMS_REGISTRY:
            defaults = setup_hparams(HPARAMS_REGISTRY['defaults'], {})
            for key in defaults:
                if key not in cfg:
                    cfg[key] = defaults[key]
                elif isinstance(defaults[key], Hyperparams) and isinstance(cfg[key], Hyperparams):
                    for subkey in defaults[key]:
                        if subkey not in cfg[key]:
                            cfg[key][subkey] = defaults[key][subkey]
        
        # Ensure model config exists
        if 'model' not in cfg:
            if 'model' in HPARAMS_REGISTRY:
                cfg['model'] = setup_hparams(HPARAMS_REGISTRY['model'], {})
            else:
                raise ValueError("'model' config not found. Please provide model configuration.")
        
        model_cfg = cfg.model
        
        # Preprocessor
        print("Initializing preprocessor...")
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=model_cfg.sample_rate,
            normalize=model_cfg.preprocessor.normalize,
            window_size=model_cfg.preprocessor.window_size,
            window_stride=model_cfg.preprocessor.window_stride,
            features=model_cfg.preprocessor.features,
            n_fft=model_cfg.preprocessor.n_fft,
            log=model_cfg.preprocessor.get('log', True),
            frame_splicing=1,
            dither=model_cfg.preprocessor.get('dither', 0.0),
            pad_to=model_cfg.preprocessor.get('pad_to', 16),
            pad_value=0.0,
        )
        
        # Quantizer
        print("Initializing quantizer...")
        self.quantizer = RandomProjectionVectorQuantizer(
            feat_in=model_cfg.preprocessor.features,
            code_dim=model_cfg.code_dim,
            num_books=model_cfg.num_books,
            num_classes=model_cfg.num_classes,
            dist_fn="l2",
            freeze=True,
            squeeze_single=model_cfg.squeeze_single,
            combine_time_steps=model_cfg.encoder.subsampling_factor,
        )
        
        # Mask processor
        print("Initializing mask_processor...")
        self.mask_processor = RandomBlockMasking(
            block_size=model_cfg.masking.block_size,
            mask_prob=model_cfg.masking.mask_prob,
            feat_in=model_cfg.preprocessor.features,
            freeze=model_cfg.masking.get('freeze', True),
            allow_overlap=model_cfg.masking.get('allow_overlap', True),
        )
        
        # Encoder
        print("Initializing encoder...")
        self.encoder = ConformerEncoder(
            feat_in=model_cfg.preprocessor.features,
            feat_out=-1,
            n_layers=model_cfg.encoder.n_layers,
            d_model=model_cfg.encoder.d_model,
            subsampling=model_cfg.encoder.subsampling,
            subsampling_factor=model_cfg.encoder.subsampling_factor,
            subsampling_conv_channels=model_cfg.encoder.subsampling_conv_channels,
            ff_expansion_factor=model_cfg.encoder.ff_expansion_factor,
            self_attention_model="rel_pos",
            n_heads=model_cfg.encoder.n_heads,
            conv_kernel_size=model_cfg.encoder.conv_kernel_size,
            dropout=model_cfg.encoder.dropout,
            dropout_pre_encoder=model_cfg.encoder.get('dropout_pre_encoder', model_cfg.encoder.dropout),
            dropout_emb=model_cfg.encoder.get('dropout_emb', 0.0),
            dropout_att=model_cfg.encoder.get('dropout_att', model_cfg.encoder.dropout),
        )
        
        # Decoder
        print("Initializing decoder...")
        self.decoder = MultiSoftmaxDecoder(
            feat_in=model_cfg.encoder.d_model,
            num_classes=model_cfg.num_classes,
            num_decoders=model_cfg.num_books,
            squeeze_single=model_cfg.squeeze_single,
            use_bias=model_cfg.decoder.get('use_bias', True),
        )
        
        # Loss
        print("Initializing loss...")
        self.loss = MultiMLMLoss(
            combine_time_steps=model_cfg.encoder.subsampling_factor,
            mask_threshold=model_cfg.loss.mask_threshold,
            num_decoders=model_cfg.num_books,
            squeeze_single=model_cfg.squeeze_single,
        )
        
        # Handle post-conv masking wrapper
        self.pre_encoder = None
        if model_cfg.get('mask_position', 'pre_conv') == "post_conv":
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
    Create optimizer from config using Optimizer hyperparams.
    
    Args:
        model: Model to optimize
        cfg: Config with cfg.Optimizer.lr, cfg.Optimizer.optimizer_name, etc.
    """
    optim_cfg = cfg.get('Optimizer', Hyperparams())
    optim_name = str(optim_cfg.get('optimizer_name', 'adamw')).lower()
    lr = float(optim_cfg.get('lr', 1e-4))
    weight_decay = float(optim_cfg.get('weight_decay', 0.0))
    beta1 = float(optim_cfg.get('beta1', 0.9))
    beta2 = float(optim_cfg.get('beta2', 0.999))
    eps = float(optim_cfg.get('eps', 1e-8))
    betas = (beta1, beta2)
    
    if optim_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay,
            eps=eps
        )
    elif optim_name == 'adam':
        return torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay,
            eps=eps
        )
    elif optim_name == 'sgd':
        momentum = float(optim_cfg.get('momentum', 0.9))
        return torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: Hyperparams, num_training_steps: int = None):
    """
    Create learning rate scheduler from config.
    
    Supports:
        - Noam annealing (from Optimizer.lr_warmup)
        - Cosine decay (from scheduler_cosine)
        - Constant (no decay)
    
    Args:
        optimizer: Optimizer instance
        cfg: Config with cfg.Optimizer or cfg.scheduler_* settings
        num_training_steps: Total training steps (for cosine decay)
    """
    optim_cfg = cfg.get('Optimizer', Hyperparams())
    
    # Check if scheduler is specified in Optimizer config
    if optim_cfg.get('lr_use_constant', False):
        return None  # No scheduler
    
    # Check for scheduler config
    sched_cfg = None
    if 'scheduler_name' in cfg:
        sched_cfg = cfg
    elif hasattr(cfg, 'scheduler_name'):
        sched_cfg = cfg
    
    if sched_cfg is None:
        # Use Noam by default if lr_warmup is set
        if optim_cfg.get('lr_warmup', 0) > 0:
            warmup_steps = int(optim_cfg.get('lr_warmup', 10000))
            # Get d_model from cfg.model.encoder.d_model
            model_cfg = cfg.get('model', Hyperparams())
            encoder_cfg = model_cfg.get('encoder', Hyperparams()) if isinstance(model_cfg, Hyperparams) else Hyperparams()
            d_model = int(encoder_cfg.get('d_model', 512))
            min_lr = float(optim_cfg.get('lr_min_scale', 0.0)) * float(optim_cfg.get('lr', 1e-4))
            if min_lr == 0:
                min_lr = 1e-6
            
            def noam_lambda(step):
                step = max(step, 1)
                scale = d_model ** (-0.5)
                lr_scale = min(step ** (-0.5), step * warmup_steps ** (-1.5))
                return max(scale * lr_scale, min_lr / optimizer.defaults['lr'])
            
            return torch.optim.lr_scheduler.LambdaLR(optimizer, noam_lambda)
        return None
    
    sched_name = str(sched_cfg.get('scheduler_name', 'noam')).lower()
    
    if sched_name in ('noamannealing', 'noam'):
        # Get d_model from cfg.model.encoder.d_model or scheduler config
        model_cfg = cfg.get('model', Hyperparams())
        encoder_cfg = model_cfg.get('encoder', Hyperparams()) if isinstance(model_cfg, Hyperparams) else Hyperparams()
        d_model = int(sched_cfg.get('d_model', encoder_cfg.get('d_model', 512)))
        warmup_steps = int(sched_cfg.get('warmup_steps', optim_cfg.get('lr_warmup', 10000)))
        min_lr = float(sched_cfg.get('min_lr', 1e-6))
        
        def noam_lambda(step):
            step = max(step, 1)
            scale = d_model ** (-0.5)
            lr_scale = min(step ** (-0.5), step * warmup_steps ** (-1.5))
            return max(scale * lr_scale, min_lr / optimizer.defaults['lr'])
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, noam_lambda)
    
    elif sched_name == 'cosine':
        warmup_steps = int(sched_cfg.get('warmup_steps', 1000))
        max_steps = num_training_steps or int(optim_cfg.get('lr_decay', 100000))
        
        def cosine_with_warmup(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)
    
    return None


def get_lr_scheduler(opt, hps):
    """
    Create learning rate scheduler based on Optimizer hyperparameters.
    
    Args:
        opt: Optimizer instance
        hps: Hyperparams config with Optimizer settings
    
    Returns:
        LambdaLR scheduler
    """
    def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return hps.Optimizer.lr_min_scale
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(hps.Optimizer.lr_min_scale, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    def lr_lambda(step):
        if hps.Optimizer.lr_use_linear_decay:
            lr_scale = hps.Optimizer.lr_scale * min(1.0, step / hps.Optimizer.lr_warmup)
            decay = max(hps.Optimizer.lr_min_scale, 1.0 - max(0.0, step - hps.Optimizer.lr_start_linear_decay) / hps.Optimizer.lr_decay)
            if decay == 0.0:
                print("Reached end of training")
            return lr_scale * decay
        
        elif hps.Optimizer.lr_use_cosine_decay:
            return _get_cosine_schedule_with_warmup_lr_lambda(
                step, 
                num_warmup_steps=hps.Optimizer.lr_warmup,
                num_training_steps=hps.Optimizer.lr_decay,
                num_cycles=0.5
            )
        
        elif hps.Optimizer.lr_use_constant:
            return 1.0
        
        elif hps.Optimizer.lr_use_noam:
            # Noam scheduler: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
            # Get d_model from model config
            d_model = hps.model.encoder.d_model if hasattr(hps, 'model') and hasattr(hps.model, 'encoder') else 512
            warmup_steps = hps.Optimizer.lr_warmup
            min_lr_scale = hps.Optimizer.lr_min_scale if hps.Optimizer.lr_min_scale > 0 else 1e-6 / hps.Optimizer.lr
            
            step = max(step, 1)  # Avoid division by zero
            scale = d_model ** (-0.5)
            lr_scale = min(step ** (-0.5), step * warmup_steps ** (-1.5))
            return max(min_lr_scale, scale * lr_scale)
        
        else:
            # Default: exponential decay with warmup
            return hps.Optimizer.lr_scale * (hps.Optimizer.lr_gamma ** (step // hps.Optimizer.lr_decay)) * min(1.0, step / hps.Optimizer.lr_warmup)
    
    shd = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return shd
