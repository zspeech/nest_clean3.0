#!/usr/bin/env python3
"""
Compare PureTorchSSLModel with EncDecDenoiseMaskedTokenPredModel.
Verifies that the pure PyTorch version produces identical outputs.
"""

import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, open_dict
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use absolute import to ensure we get nest_ssl_project's version
import sys
import os
# Add nest_ssl_project to path to ensure correct imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from models.ssl_models import EncDecDenoiseMaskedTokenPredModel

# Debug: print which module we're using
print(f"Using EncDecDenoiseMaskedTokenPredModel from: {EncDecDenoiseMaskedTokenPredModel.__module__}")
from models.ssl_model_pure_torch import PureTorchSSLModel
from data.ssl_dataset import AudioNoiseBatch
from utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy_weights(src_model, dst_model):
    """Copy weights from source to destination with automatic key renaming."""
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    
    # Create a mapping for key renaming (handles decoder vs decoder_ssl)
    def normalize_key(key):
        """Normalize key names - map decoder_ssl to decoder."""
        if key.startswith('decoder_ssl.'):
            return key.replace('decoder_ssl.', 'decoder.', 1)
        return key
    
    # Build remapped source state dict
    remapped_src_state = {}
    key_mapping = {}  # original_key -> new_key
    
    for key, value in src_state.items():
        new_key = normalize_key(key)
        if new_key in dst_state:
            remapped_src_state[new_key] = value
            if key != new_key:
                key_mapping[key] = new_key
        else:
            remapped_src_state[key] = value  # Keep original if no match
    
    src_keys = set(remapped_src_state.keys())
    dst_keys = set(dst_state.keys())
    
    only_in_src = src_keys - dst_keys
    only_in_dst = dst_keys - src_keys
    common_keys = src_keys & dst_keys
    
    shape_mismatch = []
    for key in common_keys:
        if remapped_src_state[key].shape != dst_state[key].shape:
            shape_mismatch.append(f"{key}: src={list(remapped_src_state[key].shape)} dst={list(dst_state[key].shape)}")
    
    # Print info
    if key_mapping:
        print(f"   Keys remapped ({len(key_mapping)}):")
        for orig, new in list(key_mapping.items())[:5]:
            print(f"      {orig} -> {new}")
    
    if only_in_src:
        print(f"   Keys only in original ({len(only_in_src)}):")
        for k in list(only_in_src)[:10]:
            print(f"      {k}")
    
    if only_in_dst:
        print(f"   Keys only in pure torch ({len(only_in_dst)}):")
        for k in list(only_in_dst)[:10]:
            print(f"      {k}")
    
    if shape_mismatch:
        print(f"   Shape mismatches ({len(shape_mismatch)}):")
        for k in shape_mismatch[:10]:
            print(f"      {k}")
    
    # Load remapped state dict
    dst_model.load_state_dict(remapped_src_state, strict=True)
    print(f"   Successfully loaded {len(common_keys)} keys")
    
    return len(common_keys) - len(shape_mismatch), len(only_in_src), len(only_in_dst)


def compare_tensors(name, t1, t2, rtol=1e-4, atol=1e-5):
    """Compare two tensors."""
    if t1.shape != t2.shape:
        return f"[FAIL] {name}: Shape mismatch {t1.shape} vs {t2.shape}", False
    
    max_diff = (t1.float() - t2.float()).abs().max().item()
    is_close = torch.allclose(t1.float(), t2.float(), rtol=rtol, atol=atol)
    
    status = "PASS" if is_close else "FAIL"
    return f"[{status}] {name}: max_diff={max_diff:.2e}, shape={list(t1.shape)}", is_close


def create_dummy_batch(batch_size=2, audio_len=1600000, device='cpu'):
    """Create dummy batch (10s audio to ensure masked positions exist)."""
    set_seed(42)
    return AudioNoiseBatch(
        audio=torch.randn(batch_size, audio_len, device=device),
        audio_len=torch.tensor([audio_len, audio_len], dtype=torch.int32, device=device),
        noise=torch.randn(batch_size, audio_len, device=device),
        noise_len=torch.tensor([audio_len, audio_len], dtype=torch.int32, device=device),
        noisy_audio=torch.randn(batch_size, audio_len, device=device),
        noisy_audio_len=torch.tensor([audio_len, audio_len], dtype=torch.int32, device=device),
    )


def run_comparison(config_path=None, device='cpu'):
    print("=" * 80)
    print("Comparing PureTorchSSLModel with EncDecDenoiseMaskedTokenPredModel")
    print("=" * 80)
    
    # Load config
    if config_path is None:
        config_path = os.path.join(project_root, "config", "nest_fast-conformer.yaml")
    
    print(f"\n1. Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Disable data loader setup
    with open_dict(cfg):
        if 'train_ds' in cfg.model:
            cfg.model.train_ds.defer_setup = True
        if 'validation_ds' in cfg.model:
            cfg.model.validation_ds.defer_setup = True
    
    # Mock trainer for original model
    class MockTrainer:
        world_size = 1
        global_rank = 0
        local_rank = 0
        num_devices = 1
        num_nodes = 1
        val_dataloaders = None
        test_dataloaders = None
        global_step = 0
    
    # Create original model
    print("\n2. Creating original model (EncDecDenoiseMaskedTokenPredModel)...")
    set_seed(42)
    original_model = EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=None)
    original_model._trainer = MockTrainer()
    original_model.to(device)
    original_model.eval()
    print(f"   Parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    
    # Debug: print all top-level modules
    print("   Modules in original model:")
    for name, module in original_model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print(f"      {name}: {param_count:,} params")
    
    # Check if decoder_ssl exists (should have been deleted)
    if hasattr(original_model, 'decoder_ssl'):
        print("   WARNING: decoder_ssl still exists (should have been deleted)")
        print(f"      decoder_ssl params: {sum(p.numel() for p in original_model.decoder_ssl.parameters()):,}")
    if hasattr(original_model, 'decoder'):
        print(f"   decoder exists with params: {sum(p.numel() for p in original_model.decoder.parameters()):,}")
    
    # Create pure torch model from config file
    print("\n3. Creating pure torch model (PureTorchSSLModel)...")
    set_seed(42)
    pure_torch_model = PureTorchSSLModel.from_config_file(config_path)
    pure_torch_model.to(device)
    pure_torch_model.eval()
    print(f"   Parameters: {sum(p.numel() for p in pure_torch_model.parameters()):,}")
    
    # Copy weights
    print("\n4. Copying weights from original to pure torch...")
    matched, unmatched_src, unmatched_dst = copy_weights(original_model, pure_torch_model)
    print(f"   Matched and copied: {matched}")
    print(f"   Only in original: {unmatched_src}")
    print(f"   Only in pure torch: {unmatched_dst}")
    
    # Check parameter counts per module
    if sum(p.numel() for p in original_model.parameters()) != sum(p.numel() for p in pure_torch_model.parameters()):
        print("\n   [WARN] Parameter count mismatch! Comparing by module:")
        
        def get_module_params(model, prefix=''):
            """Get parameter count per top-level module."""
            counts = {}
            for name, module in model.named_children():
                full_name = f"{prefix}{name}"
                count = sum(p.numel() for p in module.parameters())
                counts[full_name] = count
            return counts
        
        orig_counts = get_module_params(original_model)
        pure_counts = get_module_params(pure_torch_model)
        
        # Normalize module names for comparison (decoder_ssl -> decoder)
        def normalize_mod(name):
            if name == 'decoder_ssl':
                return 'decoder'
            return name
        
        normalized_orig_counts = {normalize_mod(k): v for k, v in orig_counts.items()}
        
        all_modules = set(normalized_orig_counts.keys()) | set(pure_counts.keys())
        for mod in sorted(all_modules):
            orig_c = normalized_orig_counts.get(mod, 0)
            pure_c = pure_counts.get(mod, 0)
            orig_name = 'decoder_ssl' if mod == 'decoder' and 'decoder_ssl' in orig_counts else mod
            if orig_c != pure_c:
                print(f"   {orig_name} vs {mod}: original={orig_c:,} vs pure_torch={pure_c:,} (diff={orig_c-pure_c:,})")
        
        # Check for keys only in original
        orig_state = original_model.state_dict()
        pure_state = pure_torch_model.state_dict()
        
        # Normalize keys
        def normalize_key(key):
            if key.startswith('decoder_ssl.'):
                return key.replace('decoder_ssl.', 'decoder.', 1)
            return key
        
        normalized_pure_keys = {normalize_key(k): k for k in pure_state.keys()}
        
        only_in_orig = []
        for k in orig_state.keys():
            norm_k = normalize_key(k)
            if norm_k not in normalized_pure_keys and k not in pure_state:
                only_in_orig.append(k)
        
        if only_in_orig:
            print(f"\n   Keys only in original ({len(only_in_orig)}):")
            for k in only_in_orig[:10]:
                print(f"      {k}: shape={list(orig_state[k].shape)}, params={orig_state[k].numel():,}")
    
    # Verify weights match (compare by name with normalization)
    print("\n5. Verifying weights match...")
    orig_params = dict(original_model.named_parameters())
    pure_params = dict(pure_torch_model.named_parameters())
    
    def normalize_key(key):
        if key.startswith('decoder_ssl.'):
            return key.replace('decoder_ssl.', 'decoder.', 1)
        return key
    
    all_match = True
    mismatch_count = 0
    for name in orig_params.keys():
        norm_name = normalize_key(name)
        if norm_name in pure_params:
            if not torch.equal(orig_params[name].data, pure_params[norm_name].data):
                if mismatch_count < 5:
                    print(f"   [FAIL] Weights differ: {name}")
                mismatch_count += 1
                all_match = False
        elif name in pure_params:
            if not torch.equal(orig_params[name].data, pure_params[name].data):
                if mismatch_count < 5:
                    print(f"   [FAIL] Weights differ: {name}")
                mismatch_count += 1
                all_match = False
        # Skip keys that don't exist in pure (already reported above)
    
    if mismatch_count > 5:
        print(f"   ... and {mismatch_count - 5} more mismatches")
    
    if all_match and mismatch_count == 0:
        print("   [PASS] All common weights match exactly")
    
    # Create test batch (10 seconds to ensure masked positions)
    print("\n6. Creating test batch...")
    batch = create_dummy_batch(batch_size=2, audio_len=160000, device=device)
    
    # Forward pass - original
    print("\n7. Running forward pass on original model...")
    set_seed(42)
    with torch.no_grad():
        orig_out = original_model.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noise_signal=batch.noise,
            noise_signal_length=batch.noise_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=True,
        )
    
    # Forward pass - pure torch
    print("   Running forward pass on pure torch model...")
    set_seed(42)
    with torch.no_grad():
        pure_out = pure_torch_model.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=True,
        )
    
    # Compare outputs
    print("\n8. Comparing forward outputs...")
    results = []
    pass_flags = []
    
    log_probs_orig, encoded_len_orig, masks_orig, tokens_orig = orig_out
    log_probs_pure, encoded_len_pure, masks_pure, tokens_pure = pure_out
    
    r, p = compare_tensors("log_probs", log_probs_orig.cpu(), log_probs_pure.cpu())
    results.append(r); pass_flags.append(p)
    r, p = compare_tensors("encoded_len", encoded_len_orig.cpu().float(), encoded_len_pure.cpu().float())
    results.append(r); pass_flags.append(p)
    r, p = compare_tensors("masks", masks_orig.cpu(), masks_pure.cpu())
    results.append(r); pass_flags.append(p)
    r, p = compare_tensors("tokens", tokens_orig.cpu().float(), tokens_pure.cpu().float())
    results.append(r); pass_flags.append(p)
    
    # Compare training step with FIXED masks (no randomness)
    print("\n9. Comparing training step with fixed masks...")
    
    # Create batch
    batch1 = create_dummy_batch(batch_size=2, audio_len=160000, device=device)
    
    # First, run forward WITHOUT masking to get outputs
    original_model.eval()
    pure_torch_model.eval()
    
    with torch.no_grad():
        # Get preprocessed features
        orig_processed, orig_len = original_model.preprocessor(
            input_signal=batch1.audio, length=batch1.audio_len
        )
        orig_noisy_processed, orig_noisy_len = original_model.preprocessor(
            input_signal=batch1.noisy_audio, length=batch1.noisy_audio_len
        )
        
        # Create FIXED mask (mask frames 10-50 for each sample)
        fixed_mask = torch.zeros_like(orig_noisy_processed)
        fixed_mask[:, :, 10:50] = 1.0  # Mask frames 10-50
        
        # Get tokens from clean signal
        _, orig_tokens = original_model.quantizer(input_signal=orig_processed)
        _, pure_tokens = pure_torch_model.quantizer(input_signal=orig_processed)
        
        # Apply fixed mask manually
        masked_signal = orig_noisy_processed * (1 - fixed_mask)
        
        # Encode
        orig_encoded, orig_enc_len = original_model.encoder(
            audio_signal=masked_signal, length=orig_noisy_len
        )
        pure_encoded, pure_enc_len = pure_torch_model.encoder(
            audio_signal=masked_signal, length=orig_noisy_len
        )
        
        # Decode (original model uses decoder_ssl, pure torch uses decoder)
        orig_decoder = getattr(original_model, 'decoder_ssl', original_model.decoder)
        orig_log_probs = orig_decoder(encoder_output=orig_encoded)
        pure_log_probs = pure_torch_model.decoder(encoder_output=pure_encoded)
        
        # Compute loss with fixed mask
        orig_loss = original_model.loss(
            masks=fixed_mask, decoder_outputs=orig_log_probs, 
            targets=orig_tokens, decoder_lengths=orig_enc_len
        )
        pure_loss = pure_torch_model.loss(
            masks=fixed_mask, decoder_outputs=pure_log_probs,
            targets=pure_tokens, decoder_lengths=pure_enc_len
        )
    
    print(f"   Original loss (fixed mask): {orig_loss.item():.6f}")
    print(f"   Pure torch loss (fixed mask): {pure_loss.item():.6f}")
    
    r, p = compare_tensors("training_loss_fixed_mask", orig_loss.cpu(), pure_loss.cpu())
    results.append(r)
    pass_flags.append(p)
    
    # 10. Compare backward gradients
    print("\n10. Comparing backward gradients...")
    
    # Create fresh batch for gradient comparison
    batch2 = create_dummy_batch(batch_size=2, audio_len=160000, device=device)
    
    # Zero gradients
    original_model.zero_grad()
    pure_torch_model.zero_grad()
    
    # Enable training mode
    original_model.train()
    pure_torch_model.train()
    
    # Forward pass with gradients - Original model
    set_seed(42)
    orig_processed, orig_len = original_model.preprocessor(
        input_signal=batch2.audio, length=batch2.audio_len
    )
    orig_noisy_processed, orig_noisy_len = original_model.preprocessor(
        input_signal=batch2.noisy_audio, length=batch2.noisy_audio_len
    )
    
    # Create fixed mask for gradient comparison
    fixed_mask_grad = torch.zeros_like(orig_noisy_processed)
    fixed_mask_grad[:, :, 10:50] = 1.0
    
    # Get tokens
    _, orig_tokens_grad = original_model.quantizer(input_signal=orig_processed)
    
    # Apply mask and encode
    masked_signal_grad = orig_noisy_processed * (1 - fixed_mask_grad)
    orig_encoded_grad, orig_enc_len_grad = original_model.encoder(
        audio_signal=masked_signal_grad, length=orig_noisy_len
    )
    orig_decoder = getattr(original_model, 'decoder_ssl', original_model.decoder)
    orig_log_probs_grad = orig_decoder(encoder_output=orig_encoded_grad)
    orig_loss_grad = original_model.loss(
        masks=fixed_mask_grad, decoder_outputs=orig_log_probs_grad,
        targets=orig_tokens_grad, decoder_lengths=orig_enc_len_grad
    )
    
    # Backward - Original
    orig_loss_grad.backward()
    
    # Forward pass with gradients - Pure torch model
    set_seed(42)
    pure_processed, pure_len = pure_torch_model.preprocessor(
        input_signal=batch2.audio, length=batch2.audio_len
    )
    pure_noisy_processed, pure_noisy_len = pure_torch_model.preprocessor(
        input_signal=batch2.noisy_audio, length=batch2.noisy_audio_len
    )
    
    # Get tokens
    _, pure_tokens_grad = pure_torch_model.quantizer(input_signal=pure_processed)
    
    # Apply same mask and encode
    pure_masked_signal_grad = pure_noisy_processed * (1 - fixed_mask_grad)
    pure_encoded_grad, pure_enc_len_grad = pure_torch_model.encoder(
        audio_signal=pure_masked_signal_grad, length=pure_noisy_len
    )
    pure_log_probs_grad = pure_torch_model.decoder(encoder_output=pure_encoded_grad)
    pure_loss_grad = pure_torch_model.loss(
        masks=fixed_mask_grad, decoder_outputs=pure_log_probs_grad,
        targets=pure_tokens_grad, decoder_lengths=pure_enc_len_grad
    )
    
    # Backward - Pure torch
    pure_loss_grad.backward()
    
    # Compare gradients
    print("   Comparing parameter gradients...")
    orig_params = dict(original_model.named_parameters())
    pure_params = dict(pure_torch_model.named_parameters())
    
    # Normalize key names for comparison
    def normalize_key(key):
        if key.startswith('decoder_ssl.'):
            return key.replace('decoder_ssl.', 'decoder.', 1)
        return key
    
    grad_mismatch = 0
    grad_match = 0
    max_grad_diff = 0.0
    max_grad_diff_name = ""
    
    for orig_name, orig_param in orig_params.items():
        pure_name = normalize_key(orig_name)
        if pure_name in pure_params:
            pure_param = pure_params[pure_name]
            if orig_param.grad is not None and pure_param.grad is not None:
                diff = (orig_param.grad - pure_param.grad).abs().max().item()
                if diff > max_grad_diff:
                    max_grad_diff = diff
                    max_grad_diff_name = orig_name
                if diff > 1e-4:
                    grad_mismatch += 1
                    if grad_mismatch <= 3:
                        print(f"   [FAIL] Gradient mismatch: {orig_name}, max_diff={diff:.2e}")
                else:
                    grad_match += 1
            elif orig_param.grad is None and pure_param.grad is None:
                grad_match += 1  # Both None is OK (frozen params)
    
    print(f"   Gradients matched: {grad_match}, mismatched: {grad_mismatch}")
    print(f"   Max gradient diff: {max_grad_diff:.2e} at {max_grad_diff_name}")
    
    grad_pass = grad_mismatch == 0 or max_grad_diff < 1e-3
    results.append(f"[{'PASS' if grad_pass else 'FAIL'}] gradients: matched={grad_match}, mismatched={grad_mismatch}, max_diff={max_grad_diff:.2e}")
    pass_flags.append(grad_pass)
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    for r in results:
        print(r)
    
    all_pass = all(pass_flags)
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ SUCCESS: PureTorchSSLModel is fully aligned with original!")
        print("  Note: Training loss may have small differences due to random mask generation.")
    else:
        print("✗ FAILURE: Some outputs do not match.")
    print("=" * 80)
    
    return all_pass


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    run_comparison(config_path=args.config, device=device)

