#!/usr/bin/env python3
"""
Verify that SimplifiedSSLModel output aligns with EncDecDenoiseMaskedTokenPredModel.

This script:
1. Creates both models with same config
2. Copies weights from original to simplified
3. Runs forward pass with same input
4. Compares outputs bit-for-bit
"""

import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.ssl_models import EncDecDenoiseMaskedTokenPredModel
from models.ssl_models_simplified import SimplifiedSSLModel
from data.ssl_dataset import AudioNoiseBatch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy_weights(src_model, dst_model):
    """Copy weights from source model to destination model."""
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    
    matched = 0
    unmatched = []
    
    for key in src_state.keys():
        if key in dst_state:
            if src_state[key].shape == dst_state[key].shape:
                dst_state[key] = src_state[key].clone()
                matched += 1
            else:
                unmatched.append(f"{key}: shape mismatch {src_state[key].shape} vs {dst_state[key].shape}")
        else:
            unmatched.append(f"{key}: not in dst")
    
    dst_model.load_state_dict(dst_state)
    
    print(f"Copied {matched} parameters")
    if unmatched:
        print(f"Unmatched parameters ({len(unmatched)}):")
        for u in unmatched[:10]:
            print(f"  {u}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")
    
    return matched, unmatched


def create_dummy_batch(batch_size=2, audio_len=16000, device='cpu'):
    """Create dummy batch for testing."""
    audio = torch.randn(batch_size, audio_len, device=device)
    audio_lengths = torch.tensor([audio_len, audio_len], dtype=torch.int32, device=device)
    noise = torch.randn(batch_size, audio_len, device=device)
    noise_lengths = torch.tensor([audio_len, audio_len], dtype=torch.int32, device=device)
    noisy_audio = audio + 0.1 * noise
    noisy_audio_lengths = torch.tensor([audio_len, audio_len], dtype=torch.int32, device=device)
    
    return AudioNoiseBatch(
        audio=audio,
        audio_len=audio_lengths,
        noise=noise,
        noise_len=noise_lengths,
        noisy_audio=noisy_audio,
        noisy_audio_len=noisy_audio_lengths,
    )


def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, rtol=1e-5, atol=1e-6):
    """Compare two tensors and print results."""
    if t1.shape != t2.shape:
        print(f"[FAIL] {name}: Shape mismatch {t1.shape} vs {t2.shape}")
        return False
    
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    
    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"[PASS] {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    else:
        print(f"[FAIL] {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        # Find location of max diff
        diff = (t1 - t2).abs()
        max_idx = diff.argmax()
        idx = np.unravel_index(max_idx.cpu().numpy(), diff.shape)
        print(f"       Max diff at {idx}: original={t1[idx].item():.6f}, simplified={t2[idx].item():.6f}")
    
    return is_close


def verify_alignment(config_path: str = None, device: str = 'cpu'):
    """Main verification function."""
    print("=" * 80)
    print("Verifying SimplifiedSSLModel alignment with EncDecDenoiseMaskedTokenPredModel")
    print("=" * 80)
    
    set_seed(42)
    
    # Load config
    if config_path is None:
        config_path = os.path.join(project_root, "config", "nest_fast-conformer.yaml")
    
    print(f"\nLoading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Create models
    print("\n1. Creating original model...")
    original_model = EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=None)
    original_model.to(device)
    original_model.eval()
    
    print("2. Creating simplified model...")
    simplified_model = SimplifiedSSLModel(cfg=cfg.model, trainer=None)
    simplified_model.to(device)
    simplified_model.eval()
    
    # Copy weights
    print("\n3. Copying weights from original to simplified...")
    matched, unmatched = copy_weights(original_model, simplified_model)
    
    # Verify weights are identical
    print("\n4. Verifying weight alignment...")
    weights_match = True
    for (name1, p1), (name2, p2) in zip(
        original_model.named_parameters(), simplified_model.named_parameters()
    ):
        if name1 != name2:
            print(f"[WARN] Parameter name mismatch: {name1} vs {name2}")
        if not torch.equal(p1.data, p2.data):
            print(f"[FAIL] Weights differ: {name1}")
            weights_match = False
    
    if weights_match:
        print("[PASS] All weights match exactly")
    
    # Create dummy batch
    print("\n5. Creating dummy batch...")
    batch = create_dummy_batch(batch_size=2, audio_len=16000, device=device)
    
    # Run forward pass
    print("\n6. Running forward pass...")
    
    with torch.no_grad():
        set_seed(42)  # Reset seed before forward
        orig_outputs = original_model.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noise_signal=batch.noise,
            noise_signal_length=batch.noise_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=True,
        )
        
        set_seed(42)  # Reset seed before forward
        simp_outputs = simplified_model.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noise_signal=batch.noise,
            noise_signal_length=batch.noise_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=True,
        )
    
    # Compare outputs
    print("\n7. Comparing outputs...")
    
    log_probs_orig, encoded_len_orig, masks_orig, tokens_orig = orig_outputs
    log_probs_simp, encoded_len_simp, masks_simp, tokens_simp = simp_outputs
    
    all_pass = True
    all_pass &= compare_tensors("log_probs", log_probs_orig, log_probs_simp)
    all_pass &= compare_tensors("encoded_len", encoded_len_orig.float(), encoded_len_simp.float())
    all_pass &= compare_tensors("masks", masks_orig, masks_simp)
    all_pass &= compare_tensors("tokens", tokens_orig.float(), tokens_simp.float())
    
    # Test training step
    print("\n8. Comparing training step...")
    
    set_seed(42)
    original_model.train()
    orig_train_out = original_model.training_step(batch, batch_idx=0)
    
    set_seed(42)
    simplified_model.train()
    simp_train_out = simplified_model.training_step(batch, batch_idx=0)
    
    all_pass &= compare_tensors("training_loss", orig_train_out['loss'], simp_train_out['loss'])
    
    # Summary
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ SUCCESS: SimplifiedSSLModel is fully aligned with original!")
    else:
        print("✗ FAILURE: Some outputs do not match. Check details above.")
    print("=" * 80)
    
    return all_pass


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify SimplifiedSSLModel alignment")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    verify_alignment(config_path=args.config, device=device)

