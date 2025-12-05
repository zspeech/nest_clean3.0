#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Hyperparams configuration system.

Usage:
    python test_hparams.py
"""

import sys
import torch

# Test imports
print("=" * 60)
print("Testing Hyperparams System")
print("=" * 60)

# ============================================================================
# Test 1: Import
# ============================================================================
print("\n[Test 1] Importing modules...")
try:
    from hparams import (
        Hyperparams, 
        setup_hparams, 
        get_config, 
        HPARAMS_REGISTRY,
        list_registered,
        print_config,
    )
    print("  ✓ hparams imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import hparams: {e}")
    sys.exit(1)

try:
    from ssl_model import PureTorchSSLModel, create_optimizer
    print("  ✓ ssl_model imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import ssl_model: {e}")
    sys.exit(1)

# ============================================================================
# Test 2: Hyperparams creation and access
# ============================================================================
print("\n[Test 2] Hyperparams creation and attribute access...")

# Basic Hyperparams
hp = Hyperparams(a=1, b=2, c=3)
assert hp.a == 1, "hp.a should be 1"
assert hp.b == 2, "hp.b should be 2"
assert hp['c'] == 3, "hp['c'] should be 3"
print("  ✓ Basic Hyperparams works")

# Nested Hyperparams
hp = Hyperparams(
    encoder=Hyperparams(d_model=512, n_layers=17),
    optim=Hyperparams(lr=1e-4),
)
assert hp.encoder.d_model == 512, "hp.encoder.d_model should be 512"
assert hp.encoder.n_layers == 17, "hp.encoder.n_layers should be 17"
assert hp.optim.lr == 1e-4, "hp.optim.lr should be 1e-4"
print("  ✓ Nested Hyperparams works")

# .get() method
assert hp.get('missing', 'default') == 'default', ".get() with default should work"
assert hp.encoder.get('missing', 100) == 100, "Nested .get() should work"
print("  ✓ .get() method works")

# ============================================================================
# Test 3: setup_hparams from dict
# ============================================================================
print("\n[Test 3] setup_hparams from dict...")

config = {
    'sample_rate': 16000,
    'encoder': {
        'd_model': 256,
        'n_layers': 8,
    },
    'optim': {
        'lr': 5e-5,
    },
}
cfg = setup_hparams(config, {})

assert cfg.sample_rate == 16000, "cfg.sample_rate should be 16000"
assert cfg.encoder.d_model == 256, "cfg.encoder.d_model should be 256"
assert cfg.encoder.n_layers == 8, "cfg.encoder.n_layers should be 8"
assert cfg.optim.lr == 5e-5, "cfg.optim.lr should be 5e-5"
print("  ✓ setup_hparams from dict works")

# With overrides
cfg = setup_hparams(config, {'encoder': {'d_model': 512}})
assert cfg.encoder.d_model == 512, "Override should change d_model to 512"
assert cfg.encoder.n_layers == 8, "n_layers should remain 8"
print("  ✓ setup_hparams with overrides works")

# ============================================================================
# Test 4: Registry
# ============================================================================
print("\n[Test 4] HPARAMS_REGISTRY...")

registered = list_registered()
print(f"  Registered configs: {len(registered)}")

# Check some expected configs
expected_configs = ['ssl_small', 'ssl_medium', 'ssl_large', 'ssl_xlarge', 'encoder_large', 'optimizer_adamw']
for name in expected_configs:
    assert name in registered, f"{name} should be registered"
print(f"  ✓ All expected configs are registered")

# ============================================================================
# Test 5: get_config
# ============================================================================
print("\n[Test 5] get_config...")

cfg = get_config("ssl_large")
assert cfg.encoder.d_model == 512, "ssl_large should have d_model=512"
assert cfg.encoder.n_layers == 17, "ssl_large should have n_layers=17"
assert cfg.encoder.n_heads == 8, "ssl_large should have n_heads=8"
print("  ✓ get_config('ssl_large') works")

cfg = get_config("ssl_small")
assert cfg.encoder.d_model == 176, "ssl_small should have d_model=176"
assert cfg.encoder.n_layers == 16, "ssl_small should have n_layers=16"
print("  ✓ get_config('ssl_small') works")

# With overrides
cfg = get_config("ssl_large", {"encoder": {"dropout": 0.2}})
assert cfg.encoder.dropout == 0.2, "Override should change dropout to 0.2"
assert cfg.encoder.d_model == 512, "d_model should remain 512"
print("  ✓ get_config with overrides works")

# ============================================================================
# Test 6: Model creation with config
# ============================================================================
print("\n[Test 6] Model creation with config...")

# Use small config for faster testing
cfg = get_config("ssl_small")
print(f"  Creating model with config: d_model={cfg.encoder.d_model}, n_layers={cfg.encoder.n_layers}")

try:
    model = PureTorchSSLModel(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created successfully with {num_params:,} parameters")
except Exception as e:
    print(f"  ✗ Failed to create model: {e}")
    sys.exit(1)

# Verify model components
assert hasattr(model, 'preprocessor'), "Model should have preprocessor"
assert hasattr(model, 'encoder'), "Model should have encoder"
assert hasattr(model, 'decoder'), "Model should have decoder"
assert hasattr(model, 'quantizer'), "Model should have quantizer"
assert hasattr(model, 'mask_processor'), "Model should have mask_processor"
assert hasattr(model, 'loss'), "Model should have loss"
print("  ✓ All model components exist")

# ============================================================================
# Test 7: Forward pass
# ============================================================================
print("\n[Test 7] Forward pass...")

batch_size = 2
audio_len = 16000  # 1 second

audio = torch.randn(batch_size, audio_len)
audio_lens = torch.tensor([audio_len, audio_len - 1600], dtype=torch.int32)
noisy_audio = audio + torch.randn_like(audio) * 0.1
noisy_audio_lens = audio_lens.clone()

try:
    model.eval()
    with torch.no_grad():
        outputs = model(audio, audio_lens, noisy_audio, noisy_audio_lens, apply_mask=True)
    
    assert 'loss' in outputs, "outputs should have 'loss'"
    assert 'log_probs' in outputs, "outputs should have 'log_probs'"
    assert 'masks' in outputs, "outputs should have 'masks'"
    assert 'tokens' in outputs, "outputs should have 'tokens'"
    
    print(f"  ✓ Forward pass successful")
    print(f"    - loss shape: {outputs['loss'].shape}")
    print(f"    - log_probs shape: {outputs['log_probs'].shape}")
    print(f"    - masks shape: {outputs['masks'].shape}")
    print(f"    - tokens shape: {outputs['tokens'].shape}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 8: Optimizer creation
# ============================================================================
print("\n[Test 8] Optimizer creation...")

try:
    optimizer = create_optimizer(model, cfg)
    print(f"  ✓ Optimizer created: {type(optimizer).__name__}")
    print(f"    - lr: {optimizer.defaults['lr']}")
except Exception as e:
    print(f"  ✗ Failed to create optimizer: {e}")
    sys.exit(1)

# ============================================================================
# Test 9: Backward pass
# ============================================================================
print("\n[Test 9] Backward pass...")

model.train()
optimizer.zero_grad()

try:
    outputs = model(audio, audio_lens, noisy_audio, noisy_audio_lens, apply_mask=True)
    loss = outputs['loss']
    loss.backward()
    optimizer.step()
    
    print(f"  ✓ Backward pass successful")
    print(f"    - loss value: {loss.item():.4f}")
except Exception as e:
    print(f"  ✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 10: Config access patterns
# ============================================================================
print("\n[Test 10] Config access patterns...")

cfg = get_config("ssl_large")

# Direct access
assert cfg.sample_rate == 16000
assert cfg.encoder.d_model == 512
assert cfg.encoder.n_layers == 17
print("  ✓ cfg.key.subkey works")

# .get() with default
val = cfg.get('nonexistent', 'default_value')
assert val == 'default_value'
val = cfg.encoder.get('nonexistent', 999)
assert val == 999
print("  ✓ cfg.get('key', default) works")

# Dict-style access
assert cfg['sample_rate'] == 16000
assert cfg['encoder']['d_model'] == 512
print("  ✓ cfg['key']['subkey'] works")

# Iteration
keys = list(cfg.keys())
assert 'sample_rate' in keys
assert 'encoder' in keys
print("  ✓ Iteration works")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)

print("\nQuick usage example:")
print("""
from hparams import get_config
from ssl_model import PureTorchSSLModel, create_optimizer

# Create model
cfg = get_config("ssl_large", {"optim": {"lr": 5e-5}})
model = PureTorchSSLModel(cfg).cuda()
optimizer = create_optimizer(model, cfg)

# Training
outputs = model(audio, audio_len, noisy_audio, noisy_audio_len)
outputs['loss'].backward()
optimizer.step()
""")

