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
    """Copy weights from source to destination."""
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    
    matched = 0
    for key in src_state.keys():
        if key in dst_state and src_state[key].shape == dst_state[key].shape:
            dst_state[key] = src_state[key].clone()
            matched += 1
    
    dst_model.load_state_dict(dst_state)
    return matched


def compare_tensors(name, t1, t2, rtol=1e-4, atol=1e-5):
    """Compare two tensors."""
    if t1.shape != t2.shape:
        return f"[FAIL] {name}: Shape mismatch {t1.shape} vs {t2.shape}"
    
    max_diff = (t1.float() - t2.float()).abs().max().item()
    is_close = torch.allclose(t1.float(), t2.float(), rtol=rtol, atol=atol)
    
    status = "PASS" if is_close else "FAIL"
    return f"[{status}] {name}: max_diff={max_diff:.2e}, shape={list(t1.shape)}"


def create_dummy_batch(batch_size=2, audio_len=160000, device='cpu'):
    """Create dummy batch with longer audio (10s) to ensure masked positions exist."""
    set_seed(42)
    return AudioNoiseBatch(
        audio=torch.randn(batch_size, audio_len, device=device),
        audio_len=torch.tensor([audio_len, audio_len - 10000], dtype=torch.int32, device=device),
        noise=torch.randn(batch_size, audio_len, device=device),
        noise_len=torch.tensor([audio_len, audio_len - 10000], dtype=torch.int32, device=device),
        noisy_audio=torch.randn(batch_size, audio_len, device=device),
        noisy_audio_len=torch.tensor([audio_len, audio_len - 10000], dtype=torch.int32, device=device),
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
    matched = copy_weights(original_model, pure_torch_model)
    print(f"   Copied {matched} parameters")
    
    # Verify weights match
    print("\n5. Verifying weights match...")
    all_match = True
    for (name1, p1), (name2, p2) in zip(
        original_model.named_parameters(), pure_torch_model.named_parameters()
    ):
        if not torch.equal(p1.data, p2.data):
            print(f"   [FAIL] Weights differ: {name1}")
            all_match = False
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
    
    log_probs_orig, encoded_len_orig, masks_orig, tokens_orig = orig_out
    log_probs_pure, encoded_len_pure, masks_pure, tokens_pure = pure_out
    
    results.append(compare_tensors("log_probs", log_probs_orig.cpu(), log_probs_pure.cpu()))
    results.append(compare_tensors("encoded_len", encoded_len_orig.cpu().float(), encoded_len_pure.cpu().float()))
    results.append(compare_tensors("masks", masks_orig.cpu(), masks_pure.cpu()))
    results.append(compare_tensors("tokens", tokens_orig.cpu().float(), tokens_pure.cpu().float()))
    
    # Compare training step - need to create fresh batch and set seed for each
    print("\n9. Comparing training step...")
    
    # Create fresh batch for original model
    batch1 = create_dummy_batch(batch_size=2, audio_len=160000, device=device)
    set_seed(42)
    original_model.train()
    orig_train = original_model.training_step(batch1, batch_idx=0)
    
    # Create fresh batch for pure torch model (same seed produces same batch)
    batch2 = create_dummy_batch(batch_size=2, audio_len=160000, device=device)
    set_seed(42)
    pure_torch_model.train()
    pure_train_loss = pure_torch_model.training_step(batch2)
    
    results.append(compare_tensors("training_loss", orig_train['loss'].cpu(), pure_train_loss.cpu()))
    
    # If training loss doesn't match, do detailed comparison
    if not torch.allclose(orig_train['loss'].cpu(), pure_train_loss.cpu(), rtol=1e-4, atol=1e-5):
        print("\n   Training loss mismatch - detailed comparison:")
        print(f"   Original loss: {orig_train['loss'].item():.6f}")
        print(f"   Pure torch loss: {pure_train_loss.item():.6f}")
        print(f"   Difference: {abs(orig_train['loss'].item() - pure_train_loss.item()):.2e}")
        
        # Run forward to get intermediate outputs
        set_seed(42)
        original_model.eval()
        with torch.no_grad():
            batch3 = create_dummy_batch(batch_size=2, audio_len=160000, device=device)
            set_seed(42)
            orig_log_probs, orig_enc_len, orig_masks, orig_tokens = original_model.forward(
                input_signal=batch3.audio,
                input_signal_length=batch3.audio_len,
                noise_signal=batch3.noise,
                noise_signal_length=batch3.noise_len,
                noisy_input_signal=batch3.noisy_audio,
                noisy_input_signal_length=batch3.noisy_audio_len,
                apply_mask=True,
            )
        
        set_seed(42)
        pure_torch_model.eval()
        with torch.no_grad():
            batch4 = create_dummy_batch(batch_size=2, audio_len=160000, device=device)
            set_seed(42)
            pure_log_probs, pure_enc_len, pure_masks, pure_tokens = pure_torch_model.forward(
                input_signal=batch4.audio,
                input_signal_length=batch4.audio_len,
                noisy_input_signal=batch4.noisy_audio,
                noisy_input_signal_length=batch4.noisy_audio_len,
                apply_mask=True,
            )
        
        print(f"\n   Masks comparison (with fresh seed):")
        print(f"   Original masks sum: {orig_masks.sum().item():.2f}")
        print(f"   Pure torch masks sum: {pure_masks.sum().item():.2f}")
        print(f"   Masks max diff: {(orig_masks - pure_masks).abs().max().item():.2e}")
        
        print(f"\n   Log probs comparison:")
        print(f"   Original mean: {orig_log_probs.mean().item():.6f}")
        print(f"   Pure torch mean: {pure_log_probs.mean().item():.6f}")
        print(f"   Max diff: {(orig_log_probs - pure_log_probs).abs().max().item():.2e}")
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    all_pass = True
    for r in results:
        print(r)
        if "[FAIL]" in r:
            all_pass = False
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ SUCCESS: PureTorchSSLModel is fully aligned with original!")
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

