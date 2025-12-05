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
    lr=1.0e-5,
    weight_decay=0.0,
    eps=1e-08,
    lr_min_scale=0.0,
    lr_use_linear_decay=False,
    lr_scale=1.0,
    lr_warmup=100.0,
    lr_start_linear_decay=0,
    lr_decay=10000000000.0,
    lr_use_cosine_decay=False,
    lr_use_constant=False,
    lr_use_noam=False,  # Use Noam scheduler
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
        window="hann",
        n_fft=512,
        normalize="per_feature",
        log=True,
        dither=0.0,
        pad_to=16,
        pad_value=0.0,
        frame_splicing=1,
        preemph=0.97,
        lowfreq=0,
        highfreq=None,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        exact_pad=False,
        mag_power=2.0,
        mel_norm="slaney",
        use_torchaudio=False,
    ),
    
    spec_augment=Hyperparams(
        freq_masks=0,
        time_masks=0,
        freq_width=27,
        time_width=0.05,
    ),
    
    encoder=Hyperparams(
        feat_in=80,
        n_layers=17,
        d_model=512,
        feat_out=-1,
        causal_downsampling=False,
        subsampling="dw_striding",
        subsampling_factor=8,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=256,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model="rel_pos",
        n_heads=8,
        att_context_size=[-1, -1],  # From config.yaml: [-1, -1] means unlimited context
        att_context_probs=None,
        att_context_style="regular",
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=9,
        conv_norm_type="batch_norm",
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.0,
        dropout_att=0.1,
        stochastic_depth_drop_prob=0.0,
        stochastic_depth_mode="linear",
        stochastic_depth_start_layer=1,
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        sync_max_audio_length=False,  # From config.yaml: false to prevent DDP deadlock
    ),
    
    masking=Hyperparams(
        feat_in=80,
        block_size=40,
        mask_prob=0.01,
        mask_value=None,
        freeze=True,
        allow_overlap=True,
        max_mask_ratio=0.8,
    ),
    
    decoder=Hyperparams(
        feat_in=512,
        num_classes=8192,
        num_decoders=1,
        use_bias=True,
        squeeze_single=False,
        init_mode="xavier_uniform",
    ),
    
    loss=Hyperparams(
        combine_time_steps=8,
        mask_threshold=0.8,
        num_decoders=1,
        squeeze_single=False,
    ),
    
    quantizer=Hyperparams(
        feat_in=80,
        code_dim=16,
        num_classes=8192,
        num_books=1,
        dist_fn="l2",
        time_ahead=False,
        freeze=True,
        squeeze_single=False,
        combine_time_steps=8,
    ),
    
    # Data loader configs (excluding paths)
    train_ds=Hyperparams(
        sample_rate=16000,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        use_start_end_token=False,
        trim_silence=False,
        max_duration=60.0,
        min_duration=1.0,
        drop_last=True,
        is_concat=False,
        concat_sampling_technique="temperature",
        concat_sampling_temperature=1.0,
        is_tarred=False,
        shuffle_n=2048,
        bucketing_strategy="synced_randomized",
        bucketing_batch_size=None,
        batch_augmentor=Hyperparams(
            prob=0.0,
            noise_ratio=0.5,
            min_r_speech=-5.0,
            max_r_speech=5.0,
            min_r_noise=-5.0,
            max_r_noise=20.0,
            min_mix_rate=0.5,
            max_mix_rate=0.5,
            min_num_segments=1,
            max_num_segments=1,
            min_num_speakers=1,
            max_num_speakers=1,
        ),
    ),
    
    validation_ds=Hyperparams(
        sample_rate=16000,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        use_start_end_token=False,
        max_duration=60.0,
        min_duration=1.0,
        batch_augmentor=Hyperparams(
            prob=0.0,
            noise_ratio=0.5,
            min_r_speech=-5.0,
            max_r_speech=5.0,
            min_r_noise=-5.0,
            max_r_noise=20.0,
            min_mix_rate=0.5,
            max_mix_rate=0.5,
            min_num_segments=1,
            max_num_segments=1,
            min_num_speakers=1,
            max_num_speakers=1,
        ),
    ),
    
    # Optimizer config from config.yaml (for Noam scheduler with high initial LR)
    optim=Hyperparams(
        name="adamw",
        lr=5.0,  # High initial LR for Noam scheduler
        betas=[0.9, 0.98],
        weight_decay=1e-3,
        sched=Hyperparams(
            name="NoamAnnealing",
            d_model=512,  # Will be resolved from model.encoder.d_model
            warmup_steps=25000,
            warmup_ratio=None,
            min_lr=1e-6,
        ),
    ),
)

HPARAMS_REGISTRY["model"] = model

# ============================================================================
# Trainer Configuration (from config.yaml, excluding paths)
# ============================================================================

trainer = Hyperparams(
    devices=1,
    num_nodes=1,
    max_epochs=-1,
    max_steps=500000,
    val_check_interval=1.0,
    accelerator="auto",
    strategy="auto",
    accumulate_grad_batches=1,
    gradient_clip_val=0.0,
    precision=32,
    log_every_n_steps=10,
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=1,
    sync_batchnorm=True,
    enable_checkpointing=False,
    logger=False,
    benchmark=False,
)

HPARAMS_REGISTRY["trainer"] = trainer

# ============================================================================
# Experiment Manager Configuration (from config.yaml, excluding paths)
# ============================================================================

exp_manager = Hyperparams(
    exp_dir=None,
    name="SSL-NEST-FastConformer",
    create_tensorboard_logger=True,
    create_checkpoint_callback=True,
    checkpoint_callback_params=Hyperparams(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        always_save_nemo=True,
        filename="SSL-NEST-FastConformer--{val_loss:.3f}-{step}",
    ),
    resume_if_exists=True,
    resume_ignore_no_checkpoint=True,
    create_wandb_logger=False,
    wandb_logger_kwargs=Hyperparams(
        name=None,
        project=None,
    ),
)

HPARAMS_REGISTRY["exp_manager"] = exp_manager

# ============================================================================
# Training Configuration (from config.yaml top-level, excluding paths)
# ============================================================================

training_config = Hyperparams(
    name="SSL-NEST-FastConformer",
    seed=42,
    save_steps="0,1,2,3,4",
    atol=1e-5,
    rtol=1e-5,
)

HPARAMS_REGISTRY["training_config"] = training_config

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
