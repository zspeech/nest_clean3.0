"""
Compare alignment between original NeMo SSL model and pure_torch_ssl model.
Run from the pure_torch_ssl folder or set PYTHONPATH appropriately.

Usage:
    cd pure_torch_ssl
    python compare_alignment.py --config config.yaml
    
    # Or with NeMo path:
    PYTHONPATH="/path/to/NeMo:$PYTHONPATH" python compare_alignment.py --config config.yaml
"""

import argparse
import sys
import os
import yaml
import torch
import torch.nn as nn
import random
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ssl_model import PureTorchSSLModel


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config and resolve interpolations."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    def resolve(obj, root):
        if isinstance(obj, dict):
            return {k: resolve(v, root) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve(elem, root) for elem in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            path = obj[2:-1].split('.')
            val = root
            for k in path:
                val = val.get(k) if isinstance(val, dict) else obj
            return val
        return obj
    
    return resolve(cfg, cfg)


def copy_weights(src_model, dst_model):
    """Copy weights from source to destination model with key remapping."""
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    
    # Remap decoder_ssl -> decoder
    remapped = {}
    skipped = []
    has_decoder_ssl = any(k.startswith('decoder_ssl.') for k in src_state.keys())
    
    for key, value in src_state.items():
        if key.startswith('decoder_ssl.'):
            new_key = key.replace('decoder_ssl.', 'decoder.', 1)
            remapped[new_key] = value
        elif key.startswith('decoder.') and has_decoder_ssl:
            skipped.append(key)
        else:
            remapped[key] = value
    
    # Check for missing/extra keys
    src_keys = set(remapped.keys())
    dst_keys = set(dst_state.keys())
    
    missing = dst_keys - src_keys
    extra = src_keys - dst_keys
    
    if missing:
        print(f"[WARN] Keys missing in source: {missing}")
    if extra:
        print(f"[WARN] Extra keys in source: {extra}")
    
    # Load weights
    dst_model.load_state_dict(remapped, strict=False)
    return len(src_keys & dst_keys), len(missing), len(extra)


def verify_weights(src_model, dst_model):
    """Verify weights match between models."""
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    
    has_decoder_ssl = any(k.startswith('decoder_ssl.') for k in src_state.keys())
    
    mismatches = []
    for dst_key in dst_state.keys():
        # Find corresponding source key
        if dst_key.startswith('decoder.') and has_decoder_ssl:
            src_key = dst_key.replace('decoder.', 'decoder_ssl.', 1)
        else:
            src_key = dst_key
        
        if src_key not in src_state:
            continue
            
        src_val = src_state[src_key]
        dst_val = dst_state[dst_key]
        
        if not torch.allclose(src_val, dst_val, atol=1e-6):
            diff = (src_val - dst_val).abs().max().item()
            mismatches.append((dst_key, diff))
    
    return mismatches


def compare_tensors(name, t1, t2, rtol=1e-4, atol=1e-5):
    """Compare two tensors and return result."""
    if t1 is None and t2 is None:
        return True, 0.0
    if t1 is None or t2 is None:
        return False, float('inf')
    
    t1 = t1.float().detach()
    t2 = t2.float().detach()
    
    if t1.shape != t2.shape:
        print(f"  [FAIL] {name}: shape mismatch {t1.shape} vs {t2.shape}")
        return False, float('inf')
    
    max_diff = (t1 - t2).abs().max().item()
    passed = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}: max_diff={max_diff:.2e}")
    return passed, max_diff


def run_comparison(config_path: str, device: str = 'cuda'):
    """Run full comparison between original and pure torch models."""
    print("=" * 80)
    print("Comparing PureTorchSSLModel with Original NeMo SSL Model")
    print("=" * 80)
    
    # 1. Load config
    print(f"\n1. Loading config from: {config_path}")
    cfg = load_yaml_config(config_path)
    
    # 2. Try to import original model
    print("\n2. Creating original model...")
    try:
        # Try multiple possible locations for the original model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        # Possible paths where models/ssl_models.py might be
        possible_paths = [
            parent_dir,  # ../
            os.path.join(parent_dir, 'nest_ssl_v2'),  # ../nest_ssl_v2
            os.path.join(parent_dir, 'nest_ssl_project'),  # ../nest_ssl_project
            os.getcwd(),  # current working directory
        ]
        
        model_found = False
        for path in possible_paths:
            models_path = os.path.join(path, 'models', 'ssl_models.py')
            if os.path.exists(models_path):
                if path not in sys.path:
                    sys.path.insert(0, path)
                print(f"   Found original model in: {path}")
                model_found = True
                break
        
        if not model_found:
            print(f"   Searched paths: {possible_paths}")
            raise ImportError("Could not find models/ssl_models.py")
        
        from models.ssl_models import EncDecDenoiseMaskedTokenPredModel
        from omegaconf import OmegaConf, open_dict
        
        # Convert dict to OmegaConf - pass only the model section
        omega_cfg = OmegaConf.create(cfg)
        
        # Get model config section
        if 'model' in omega_cfg:
            model_cfg = omega_cfg.model
        else:
            model_cfg = omega_cfg
        
        # Disable data loaders
        with open_dict(model_cfg):
            if 'train_ds' in model_cfg:
                model_cfg.train_ds = None
            if 'validation_ds' in model_cfg:
                model_cfg.validation_ds = None
            if 'test_ds' in model_cfg:
                model_cfg.test_ds = None
        
        set_seed(42)
        original_model = EncDecDenoiseMaskedTokenPredModel(cfg=model_cfg)
        original_model = original_model.to(device)
        original_model.eval()
        print(f"   Loaded successfully. Parameters: {sum(p.numel() for p in original_model.parameters()):,}")
        
    except Exception as e:
        print(f"   [ERROR] Could not load original model: {e}")
        print("   Make sure nest_ssl_project is in PYTHONPATH")
        return False
    
    # 3. Create pure torch model
    print("\n3. Creating pure torch model...")
    set_seed(42)
    pure_model = PureTorchSSLModel.from_config_file(config_path)
    pure_model = pure_model.to(device)
    pure_model.eval()
    print(f"   Parameters: {sum(p.numel() for p in pure_model.parameters()):,}")
    
    # 4. Copy weights
    print("\n4. Copying weights from original to pure torch...")
    matched, missing, extra = copy_weights(original_model, pure_model)
    print(f"   Matched: {matched}, Missing: {missing}, Extra: {extra}")
    
    # 5. Verify weights
    print("\n5. Verifying weights match...")
    mismatches = verify_weights(original_model, pure_model)
    if mismatches:
        print(f"   [FAIL] {len(mismatches)} weight mismatches:")
        for name, diff in mismatches[:5]:
            print(f"      {name}: max_diff={diff:.2e}")
    else:
        print("   [PASS] All weights match")
    
    # 6. Create test batch
    print("\n6. Creating test batch...")
    batch_size = 2
    audio_len = 160000  # 10 seconds at 16kHz
    
    set_seed(42)
    audio = torch.randn(batch_size, audio_len, device=device)
    audio_lens = torch.tensor([audio_len, audio_len - 16000], device=device, dtype=torch.int32)
    noisy_audio = audio + torch.randn_like(audio) * 0.1
    noisy_audio_lens = audio_lens.clone()
    
    # 7. Compare forward pass
    print("\n7. Comparing forward pass...")
    results = {}
    
    set_seed(42)
    with torch.no_grad():
        # Original model forward
        orig_processed, orig_len = original_model.preprocessor(input_signal=audio, length=audio_lens)
        orig_noisy, orig_noisy_len = original_model.preprocessor(input_signal=noisy_audio, length=noisy_audio_lens)
        _, orig_tokens = original_model.quantizer(input_signal=orig_processed)
        orig_masked, orig_masks = original_model.mask_processor(input_feats=orig_noisy, input_lengths=orig_noisy_len)
        orig_encoded, orig_enc_len = original_model.encoder(audio_signal=orig_masked, length=orig_noisy_len)
        
        # Get decoder (decoder_ssl or decoder)
        orig_decoder = getattr(original_model, 'decoder_ssl', original_model.decoder)
        orig_log_probs = orig_decoder(encoder_output=orig_encoded)
        
        # Pure torch forward
        set_seed(42)
        pure_processed, pure_len = pure_model.preprocessor(input_signal=audio, length=audio_lens)
        pure_noisy, pure_noisy_len = pure_model.preprocessor(input_signal=noisy_audio, length=noisy_audio_lens)
        _, pure_tokens = pure_model.quantizer(input_signal=pure_processed)
        pure_masked, pure_masks = pure_model.mask_processor(input_feats=pure_noisy, input_lengths=pure_noisy_len)
        pure_encoded, pure_enc_len = pure_model.encoder(audio_signal=pure_masked, length=pure_noisy_len)
        pure_log_probs = pure_model.decoder(encoder_output=pure_encoded)
    
    results['preprocessor'], _ = compare_tensors('preprocessor', orig_processed, pure_processed)
    results['tokens'], _ = compare_tensors('tokens', orig_tokens, pure_tokens)
    results['masks'], _ = compare_tensors('masks', orig_masks, pure_masks)
    results['encoded'], _ = compare_tensors('encoded', orig_encoded, pure_encoded)
    results['log_probs'], _ = compare_tensors('log_probs', orig_log_probs, pure_log_probs)
    
    # 8. Compare training loss with fixed mask
    print("\n8. Comparing training loss (fixed mask)...")
    
    # Create fixed mask
    fixed_mask = torch.zeros_like(orig_masks)
    fixed_mask[:, :, 80:240] = 1.0  # Mask frames 80-240
    
    set_seed(42)
    with torch.no_grad():
        # Recompute with fixed mask
        orig_masked_fixed = orig_noisy.clone()
        orig_masked_fixed = orig_masked_fixed * (1 - fixed_mask) + \
            original_model.mask_processor.mask_embedding.view(1, -1, 1) * fixed_mask
        
        pure_masked_fixed = pure_noisy.clone()
        pure_masked_fixed = pure_masked_fixed * (1 - fixed_mask) + \
            pure_model.mask_processor.mask_embedding.view(1, -1, 1) * fixed_mask
        
        orig_enc_fixed, _ = original_model.encoder(audio_signal=orig_masked_fixed, length=orig_noisy_len)
        pure_enc_fixed, _ = pure_model.encoder(audio_signal=pure_masked_fixed, length=pure_noisy_len)
        
        orig_logp_fixed = orig_decoder(encoder_output=orig_enc_fixed)
        pure_logp_fixed = pure_model.decoder(encoder_output=pure_enc_fixed)
        
        orig_loss = original_model.loss(masks=fixed_mask, decoder_outputs=orig_logp_fixed, targets=orig_tokens)
        pure_loss = pure_model.loss(masks=fixed_mask, decoder_outputs=pure_logp_fixed, targets=pure_tokens)
    
    results['loss'], loss_diff = compare_tensors('training_loss', orig_loss, pure_loss, atol=1e-4)
    print(f"   Original loss: {orig_loss.item():.6f}")
    print(f"   Pure torch loss: {pure_loss.item():.6f}")
    
    # 9. Compare gradients
    print("\n9. Comparing backward gradients...")
    
    # Fresh forward for gradient computation
    original_model.train()
    pure_model.train()
    original_model.zero_grad()
    pure_model.zero_grad()
    
    set_seed(42)
    # Original
    orig_proc, orig_len = original_model.preprocessor(input_signal=audio, length=audio_lens)
    orig_noisy, _ = original_model.preprocessor(input_signal=noisy_audio, length=noisy_audio_lens)
    _, orig_tok = original_model.quantizer(input_signal=orig_proc)
    orig_masked = orig_noisy * (1 - fixed_mask) + original_model.mask_processor.mask_embedding.view(1, -1, 1) * fixed_mask
    orig_enc, _ = original_model.encoder(audio_signal=orig_masked, length=orig_noisy_len)
    orig_logp = orig_decoder(encoder_output=orig_enc)
    orig_loss = original_model.loss(masks=fixed_mask, decoder_outputs=orig_logp, targets=orig_tok)
    orig_loss.backward()
    
    set_seed(42)
    # Pure torch
    pure_proc, pure_len = pure_model.preprocessor(input_signal=audio, length=audio_lens)
    pure_noisy, _ = pure_model.preprocessor(input_signal=noisy_audio, length=noisy_audio_lens)
    _, pure_tok = pure_model.quantizer(input_signal=pure_proc)
    pure_masked = pure_noisy * (1 - fixed_mask) + pure_model.mask_processor.mask_embedding.view(1, -1, 1) * fixed_mask
    pure_enc, _ = pure_model.encoder(audio_signal=pure_masked, length=pure_noisy_len)
    pure_logp = pure_model.decoder(encoder_output=pure_enc)
    pure_loss = pure_model.loss(masks=fixed_mask, decoder_outputs=pure_logp, targets=pure_tok)
    pure_loss.backward()
    
    # Compare gradients
    orig_grads = {k: v.grad for k, v in original_model.named_parameters() if v.grad is not None}
    pure_grads = {k: v.grad for k, v in pure_model.named_parameters() if v.grad is not None}
    
    has_decoder_ssl = any(k.startswith('decoder_ssl.') for k in orig_grads.keys())
    
    grad_matched = 0
    grad_mismatched = 0
    max_grad_diff = 0.0
    max_grad_name = ""
    
    for pure_key, pure_grad in pure_grads.items():
        if pure_key.startswith('decoder.') and has_decoder_ssl:
            orig_key = pure_key.replace('decoder.', 'decoder_ssl.', 1)
        else:
            orig_key = pure_key
        
        if orig_key not in orig_grads:
            continue
        
        orig_grad = orig_grads[orig_key]
        diff = (orig_grad - pure_grad).abs().max().item()
        
        if diff < 1e-4:
            grad_matched += 1
        else:
            grad_mismatched += 1
            if diff > max_grad_diff:
                max_grad_diff = diff
                max_grad_name = pure_key
    
    if grad_mismatched == 0:
        print(f"   [PASS] All {grad_matched} gradients match")
        results['gradients'] = True
    else:
        print(f"   [FAIL] {grad_mismatched}/{grad_matched + grad_mismatched} gradients mismatch")
        print(f"   Max diff: {max_grad_diff:.2e} at {max_grad_name}")
        results['gradients'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
    
    print("\n" + ("✓ All checks passed!" if all_passed else "✗ Some checks failed."))
    return all_passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare pure_torch_ssl with original NeMo model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    success = run_comparison(args.config, args.device)
    sys.exit(0 if success else 1)

