# Copyright (c) 2025. All rights reserved.
#
# Hyperparameters Registry for SSL Model
#
# Usage:
#     from hparams import setup_hparams, HPARAMS_REGISTRY
#     
#     # Use predefined config
#     cfg = setup_hparams(HPARAMS_REGISTRY["ssl_large"], {})
#     
#     # Or combine multiple configs with overrides
#     cfg = setup_hparams({
#         **HPARAMS_REGISTRY["ssl_large"],
#         **HPARAMS_REGISTRY["optimizer_adamw"],
#     }, {"encoder": {"d_model": 768}})

from typing import Dict, Any

__all__ = ['Hyperparams', 'setup_hparams', 'HPARAMS_REGISTRY', 'register_hparams']


# ============================================================================
# Hyperparams Class
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
    
    def get(self, key, default=None):
        """Get value with default."""
        return super().get(key, default)


def setup_hparams(config: dict, overrides: dict = None) -> Hyperparams:
    """
    Convert a nested dict config into Hyperparams with attribute access.
    
    Args:
        config: Dict config (can be nested)
        overrides: Optional dict of overrides (can be nested)
    
    Returns:
        Hyperparams object with cfg.key.subkey access
    """
    H = Hyperparams()
    
    for k, v in config.items():
        if isinstance(v, dict):
            H[k] = setup_hparams(v, {})
        else:
            H[k] = v
    
    # Apply overrides
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and k in H and isinstance(H[k], Hyperparams):
                # Merge nested dicts
                for sub_k, sub_v in v.items():
                    H[k][sub_k] = sub_v
            else:
                H[k] = v
    
    return H


# ============================================================================
# Registry
# ============================================================================

HPARAMS_REGISTRY: Dict[str, dict] = {}


def register_hparams(name: str, hparams: dict):
    """Register a hyperparameter set."""
    HPARAMS_REGISTRY[name] = hparams


# ============================================================================
# Default Configurations
# ============================================================================

# Base defaults
register_hparams("defaults", {
    "sample_rate": 16000,
    "num_classes": 8192,
    "num_books": 1,
    "code_dim": 16,
    "squeeze_single": False,
    "mask_position": "pre_conv",
})

# Preprocessor configs
register_hparams("preprocessor_default", {
    "preprocessor": {
        "features": 80,
        "window_size": 0.025,
        "window_stride": 0.01,
        "n_fft": 512,
        "normalize": "per_feature",
        "log": True,
        "dither": 0.0,
        "pad_to": 16,
    }
})

# Masking configs
register_hparams("masking_default", {
    "masking": {
        "block_size": 40,
        "mask_prob": 0.01,
        "freeze": True,
        "allow_overlap": True,
    }
})

# Loss configs
register_hparams("loss_default", {
    "loss": {
        "mask_threshold": 0.8,
    }
})

# Decoder configs
register_hparams("decoder_default", {
    "decoder": {
        "use_bias": True,
    }
})

# ============================================================================
# Encoder Variants (based on FastConformer)
# ============================================================================

# Small (14M params)
register_hparams("encoder_small", {
    "encoder": {
        "n_layers": 16,
        "d_model": 176,
        "n_heads": 4,
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "ff_expansion_factor": 4,
        "conv_kernel_size": 9,
        "dropout": 0.1,
        "dropout_pre_encoder": 0.1,
        "dropout_emb": 0.0,
        "dropout_att": 0.1,
        "use_bias": True,
        "xscaling": True,
    }
})

# Medium (32M params)
register_hparams("encoder_medium", {
    "encoder": {
        "n_layers": 16,
        "d_model": 256,
        "n_heads": 4,
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "ff_expansion_factor": 4,
        "conv_kernel_size": 9,
        "dropout": 0.1,
        "dropout_pre_encoder": 0.1,
        "dropout_emb": 0.0,
        "dropout_att": 0.1,
        "use_bias": True,
        "xscaling": True,
    }
})

# Large (120M params)
register_hparams("encoder_large", {
    "encoder": {
        "n_layers": 17,
        "d_model": 512,
        "n_heads": 8,
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "ff_expansion_factor": 4,
        "conv_kernel_size": 9,
        "dropout": 0.1,
        "dropout_pre_encoder": 0.1,
        "dropout_emb": 0.0,
        "dropout_att": 0.1,
        "use_bias": True,
        "xscaling": True,
    }
})

# XLarge (616M params)
register_hparams("encoder_xlarge", {
    "encoder": {
        "n_layers": 24,
        "d_model": 1024,
        "n_heads": 8,
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "ff_expansion_factor": 4,
        "conv_kernel_size": 9,
        "dropout": 0.1,
        "dropout_pre_encoder": 0.1,
        "dropout_emb": 0.0,
        "dropout_att": 0.1,
        "use_bias": False,
        "xscaling": False,
    }
})

# XXLarge (1.2B params)
register_hparams("encoder_xxlarge", {
    "encoder": {
        "n_layers": 42,
        "d_model": 1024,
        "n_heads": 8,
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "ff_expansion_factor": 4,
        "conv_kernel_size": 5,
        "dropout": 0.1,
        "dropout_pre_encoder": 0.1,
        "dropout_emb": 0.0,
        "dropout_att": 0.1,
        "use_bias": False,
        "xscaling": False,
    }
})

# ============================================================================
# Optimizer Configs
# ============================================================================

register_hparams("optimizer_adamw", {
    "optim": {
        "name": "adamw",
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 1e-3,
        "eps": 1e-8,
    }
})

register_hparams("optimizer_adam", {
    "optim": {
        "name": "adam",
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "eps": 1e-8,
    }
})

register_hparams("optimizer_sgd", {
    "optim": {
        "name": "sgd",
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 1e-4,
    }
})

# ============================================================================
# Scheduler Configs
# ============================================================================

register_hparams("scheduler_noam", {
    "optim": {
        "sched": {
            "name": "noam",
            "warmup_steps": 10000,
            "min_lr": 1e-6,
        }
    }
})

register_hparams("scheduler_cosine", {
    "optim": {
        "sched": {
            "name": "cosine",
            "warmup_steps": 1000,
        }
    }
})

register_hparams("scheduler_constant", {
    "optim": {
        "sched": {
            "name": "constant",
        }
    }
})

# ============================================================================
# Combined Presets (Full model configs)
# ============================================================================

def _merge_dicts(*dicts):
    """Recursively merge multiple dicts."""
    result = {}
    for d in dicts:
        for k, v in d.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = _merge_dicts(result[k], v)
            else:
                result[k] = v
    return result


# SSL Small (14M)
register_hparams("ssl_small", _merge_dicts(
    HPARAMS_REGISTRY["defaults"],
    HPARAMS_REGISTRY["preprocessor_default"],
    HPARAMS_REGISTRY["encoder_small"],
    HPARAMS_REGISTRY["masking_default"],
    HPARAMS_REGISTRY["decoder_default"],
    HPARAMS_REGISTRY["loss_default"],
    HPARAMS_REGISTRY["optimizer_adamw"],
    HPARAMS_REGISTRY["scheduler_noam"],
))

# SSL Medium (32M)
register_hparams("ssl_medium", _merge_dicts(
    HPARAMS_REGISTRY["defaults"],
    HPARAMS_REGISTRY["preprocessor_default"],
    HPARAMS_REGISTRY["encoder_medium"],
    HPARAMS_REGISTRY["masking_default"],
    HPARAMS_REGISTRY["decoder_default"],
    HPARAMS_REGISTRY["loss_default"],
    HPARAMS_REGISTRY["optimizer_adamw"],
    HPARAMS_REGISTRY["scheduler_noam"],
))

# SSL Large (120M) - Default
register_hparams("ssl_large", _merge_dicts(
    HPARAMS_REGISTRY["defaults"],
    HPARAMS_REGISTRY["preprocessor_default"],
    HPARAMS_REGISTRY["encoder_large"],
    HPARAMS_REGISTRY["masking_default"],
    HPARAMS_REGISTRY["decoder_default"],
    HPARAMS_REGISTRY["loss_default"],
    HPARAMS_REGISTRY["optimizer_adamw"],
    HPARAMS_REGISTRY["scheduler_noam"],
))

# SSL XLarge (616M)
register_hparams("ssl_xlarge", _merge_dicts(
    HPARAMS_REGISTRY["defaults"],
    HPARAMS_REGISTRY["preprocessor_default"],
    HPARAMS_REGISTRY["encoder_xlarge"],
    HPARAMS_REGISTRY["masking_default"],
    HPARAMS_REGISTRY["decoder_default"],
    HPARAMS_REGISTRY["loss_default"],
    HPARAMS_REGISTRY["optimizer_adamw"],
    HPARAMS_REGISTRY["scheduler_noam"],
))

# SSL XXLarge (1.2B)
register_hparams("ssl_xxlarge", _merge_dicts(
    HPARAMS_REGISTRY["defaults"],
    HPARAMS_REGISTRY["preprocessor_default"],
    HPARAMS_REGISTRY["encoder_xxlarge"],
    HPARAMS_REGISTRY["masking_default"],
    HPARAMS_REGISTRY["decoder_default"],
    HPARAMS_REGISTRY["loss_default"],
    HPARAMS_REGISTRY["optimizer_adamw"],
    HPARAMS_REGISTRY["scheduler_noam"],
))


# ============================================================================
# Utility Functions
# ============================================================================

def list_registered() -> list:
    """List all registered hyperparameter sets."""
    return list(HPARAMS_REGISTRY.keys())


def get_config(name: str, overrides: dict = None) -> Hyperparams:
    """
    Get a registered config with optional overrides.
    
    Args:
        name: Name of registered config (e.g., "ssl_large")
        overrides: Optional dict of overrides
    
    Returns:
        Hyperparams config
    
    Example:
        cfg = get_config("ssl_large", {"encoder": {"dropout": 0.2}})
    """
    if name not in HPARAMS_REGISTRY:
        available = list_registered()
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    
    return setup_hparams(HPARAMS_REGISTRY[name], overrides or {})


def print_config(cfg: Hyperparams, indent: int = 0):
    """Pretty print a config."""
    prefix = "  " * indent
    for k, v in cfg.items():
        if isinstance(v, Hyperparams):
            print(f"{prefix}{k}:")
            print_config(v, indent + 1)
        else:
            print(f"{prefix}{k}: {v}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Available configs:")
    for name in list_registered():
        print(f"  - {name}")
    
    print("\n" + "=" * 60)
    print("ssl_large config:")
    print("=" * 60)
    cfg = get_config("ssl_large")
    print_config(cfg)
    
    print("\n" + "=" * 60)
    print("With overrides:")
    print("=" * 60)
    cfg = get_config("ssl_large", {
        "encoder": {"dropout": 0.2, "n_layers": 24},
        "optim": {"lr": 5e-5},
    })
    print(f"cfg.encoder.dropout = {cfg.encoder.dropout}")
    print(f"cfg.encoder.n_layers = {cfg.encoder.n_layers}")
    print(f"cfg.optim.lr = {cfg.optim.lr}")

