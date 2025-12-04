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

from models.ssl_models import EncDecDenoiseMaskedTokenPredModel
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
    """Copy weights from source to destination using direct load_state_dict."""
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    
    # Check key compatibility first
    src_keys = set(src_state.keys())
    dst_keys = set(dst_state.keys())
    
    only_in_src = src_keys - dst_keys
    only_in_dst = dst_keys - src_keys
    common_keys = src_keys & dst_keys
    
    shape_mismatch = []
    for key in common_keys:
        if src_state[key].shape != dst_state[key].shape:
            shape_mismatch.append(f"{key}: src={list(src_state[key].shape)} dst={list(dst_state[key].shape)}")
    
    # Print detailed info
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
    
    # Directly load state dict
    dst_model.load_state_dict(src_state, strict=True)
    
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
    
    # Verify weights match (compare by name, not by position)
    print("\n5. Verifying weights match...")
    orig_params = dict(original_model.named_parameters())
    pure_params = dict(pure_torch_model.named_parameters())
    
    all_match = True
    mismatch_count = 0
    for name in orig_params.keys():
        if name in pure_params:
            if not torch.equal(orig_params[name].data, pure_params[name].data):
                if mismatch_count < 5:  # Only print first 5 mismatches
                    print(f"   [FAIL] Weights differ: {name}")
                mismatch_count += 1
                all_match = False
        else:
            print(f"   [WARN] Key not in pure torch: {name}")
    
    if mismatch_count > 5:
        print(f"   ... and {mismatch_count - 5} more mismatches")
    
    if all_match:
        print("   [PASS] All weights match exactly")
    
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
        
        # Decode
        orig_log_probs = original_model.decoder(encoder_output=orig_encoded)
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

