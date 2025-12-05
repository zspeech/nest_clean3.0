# Copyright (c) 2025. All rights reserved.
#
# Hyperparameters Registry for SSL Model
#
# Usage:
#     from hparams import setup_hparams, HPARAMS_REGISTRY
#     
#     cfg = setup_hparams(HPARAMS_REGISTRY["defaults"], {})
#     print(cfg.model.encoder.d_model)  # Access via cfg.model.encoder.d_model

import os.path as osp

__all__ = ['Hyperparams', 'setup_hparams', 'HPARAMS_REGISTRY', 'register_hparams']


# ============================================================================
# Hyperparams Class
# ============================================================================

class Hyperparams(dict):
    """Dict subclass that allows attribute-style access (cfg.key)."""
    
    def __getattr__(self, attr):
        return self[attr]
    
    def __setattr__(self, attr, value):
        self[attr] = value


# ============================================================================
# Registry
# ============================================================================

HPARAMS_REGISTRY = {}


def register_hparams(name: str, hparams: dict):
    """Register a hyperparameter set."""
    HPARAMS_REGISTRY[name] = hparams


def setup_hparams(hparam_set_names, kwargs):
    """
    Setup hyperparameters from registry entries and kwargs.
    
    Args:
        hparam_set_names: Dict of hyperparameters or registry key names
        kwargs: Additional overrides
    
    Returns:
        Hyperparams object
    """
    H = Hyperparams()
    
    # If hparam_set_names is a dict, use it directly
    if isinstance(hparam_set_names, dict):
        for k, v in hparam_set_names.items():
            if isinstance(v, dict):
                tmp = Hyperparams()
                tmp.update(v)
                H[k] = tmp
            else:
                H[k] = v
    else:
        # If it's a string or list, look up in registry
        if isinstance(hparam_set_names, str):
            hparam_set_names = hparam_set_names.split(",")
        
        if isinstance(hparam_set_names, (list, tuple)):
            for name in hparam_set_names:
                name = name.strip()
                if name in HPARAMS_REGISTRY:
                    hps = HPARAMS_REGISTRY[name]
                    for k, v in hps.items():
                        if isinstance(v, dict):
                            if k not in H:
                                H[k] = Hyperparams()
                            H[k].update(v)
                        else:
                            H[k] = v
    
    # Apply kwargs overrides
    for k, v in kwargs.items():
        if isinstance(v, dict):
            if k not in H:
                H[k] = Hyperparams()
            H[k].update(v)
        else:
            H[k] = v
    
    return H


# ============================================================================
# Default Configurations
# ============================================================================

# Base defaults
defaults = Hyperparams(
    project="ssl_nest",
    train=True,
    finetune=False,
    fsdp=False,
    fp16_opt=False,
    fp16=False,
    ema=False,
    cpu_ema=True,
    cpu_ema_freq=100,
    ema_fused=False,
    param_dtype="fp32",
    reduce_dtype="fp32",
    buffer_dtype="fp32",
    label_rate=25,
    fp16_loss_scale=None,
    fp16_scale_window=1000.0,
    restore_prior=None,
    mu=None,
    local_logdir="logs",
    name="SSL-NEST",
    curr_epoch=-1,
    epochs=1000000,
    grad_accum_iters=1,
    iters_before_update=1,
    log_steps=100,
    save=True,
    save_iters=10000,
)

HPARAMS_REGISTRY["defaults"] = defaults

# Utils
utils = Hyperparams(
    seed=42,
    num_threads=1,
)

HPARAMS_REGISTRY["utils"] = utils

# Data
data = Hyperparams(
    sr=16000,
    min_audio_length=1.0,
    split_by_rank=False,
    bs=8,
    nworkers=8,
    train_data_path=None,
    train_dict_path=None,
    val_data_path=None,
    val_dict_path=None,
    test_data_path=None,
    test_dict_path=None,
)

HPARAMS_REGISTRY["data"] = data

# CTC Loss (for compatibility, not used in SSL)
ctc_loss = Hyperparams(
    blank=0,
    vocab_size=8192,
)

HPARAMS_REGISTRY["ctc_loss"] = ctc_loss

# Models
Models = Hyperparams(
    SSLModel=None,  # Path to SSL model checkpoint
)

HPARAMS_REGISTRY["Models"] = Models

# Distributed
Distributed = Hyperparams(
    bucket=128,
)

HPARAMS_REGISTRY["Distributed"] = Distributed

# Optimizer
Optimizer = Hyperparams(
    beta1=0.9,
    beta2=0.999,
    lr=1.0e-4,
    weight_decay=0.0,
    eps=1e-08,
    lr_min_scale=0.0,
    lr_use_linear_decay=False,
    lr_scale=1.0,
    lr_warmup=10000.0,
    lr_start_linear_decay=0,
    lr_decay=10000000000.0,
    lr_use_cosine_decay=False,
    lr_use_constant=False,
    lr_gamma=1.0,
    ignore_grad_norm=0,
    optimizer_name="adamw",  # adamw, adam, sgd
)

HPARAMS_REGISTRY["Optimizer"] = Optimizer

# ============================================================================
# Model Configuration (SSL Model - 120M)
# ============================================================================

model = Hyperparams(
    sample_rate=16000,
    num_classes=8192,
    num_books=1,
    code_dim=16,
    squeeze_single=False,
    mask_position="pre_conv",
    
    preprocessor=Hyperparams(
        features=80,
        window_size=0.025,
        window_stride=0.01,
        n_fft=512,
        normalize="per_feature",
        log=True,
        dither=0.0,
        pad_to=16,
    ),
    
    encoder=Hyperparams(
        n_layers=17,
        d_model=512,
        n_heads=8,
        subsampling="dw_striding",
        subsampling_factor=8,
        subsampling_conv_channels=256,
        ff_expansion_factor=4,
        conv_kernel_size=9,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.0,
        dropout_att=0.1,
        use_bias=True,
        xscaling=True,
    ),
    
    masking=Hyperparams(
        block_size=40,
        mask_prob=0.01,
        freeze=True,
        allow_overlap=True,
    ),
    
    decoder=Hyperparams(
        use_bias=True,
    ),
    
    loss=Hyperparams(
        mask_threshold=0.8,
    ),
)

HPARAMS_REGISTRY["model"] = model

# ============================================================================
# Scheduler Configurations
# ============================================================================

# Noam Annealing Scheduler
scheduler_noam = Hyperparams(
    scheduler_name="noam",
    d_model=None,  # Will be set from model.encoder.d_model
    warmup_steps=10000,
    min_lr=1e-6,
)

HPARAMS_REGISTRY["scheduler_noam"] = scheduler_noam

# Cosine Annealing Scheduler
scheduler_cosine = Hyperparams(
    scheduler_name="cosine",
    warmup_steps=1000,
    max_steps=100000,
)

HPARAMS_REGISTRY["scheduler_cosine"] = scheduler_cosine

# Constant Scheduler (no decay)
scheduler_constant = Hyperparams(
    scheduler_name="constant",
)

HPARAMS_REGISTRY["scheduler_constant"] = scheduler_constant

# ============================================================================
# Utility Functions
# ============================================================================

def get_config(name: str, overrides: dict = None) -> Hyperparams:
    """
    Get a registered config with optional overrides.
    
    Args:
        name: Name of registered config (e.g., "defaults")
        overrides: Optional dict of overrides
    
    Returns:
        Hyperparams config
    
    Example:
        cfg = get_config("defaults", {"Optimizer": {"lr": 5e-5}})
        # Access: cfg.model.encoder.d_model
    """
    if name not in HPARAMS_REGISTRY:
        available = list(HPARAMS_REGISTRY.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    
    return setup_hparams(HPARAMS_REGISTRY[name], overrides or {})


def list_registered() -> list:
    """List all registered hyperparameter sets."""
    return list(HPARAMS_REGISTRY.keys())


def print_config(cfg: Hyperparams, indent: int = 0):
    """Pretty print a config."""
    prefix = "  " * indent
    for k, v in cfg.items():
        if isinstance(v, Hyperparams):
            print(f"{prefix}{k}:")
            print_config(v, indent + 1)
        else:
            print(f"{prefix}{k}: {v}")
