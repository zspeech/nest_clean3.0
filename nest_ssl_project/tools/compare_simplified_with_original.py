#!/usr/bin/env python3
"""
Compare SimplifiedSSLModel with EncDecDenoiseMaskedTokenPredModel.
This script runs both models with the same input and compares all intermediate outputs.
"""

import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import pickle
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.ssl_models import EncDecDenoiseMaskedTokenPredModel
from models.ssl_models_simplified import SimplifiedSSLModel
from data.ssl_dataset import AudioNoiseBatch
from utils.logging import get_logger

logger = get_logger(__name__)


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
    for key in src_state.keys():
        if key in dst_state and src_state[key].shape == dst_state[key].shape:
            dst_state[key] = src_state[key].clone()
            matched += 1
    
    dst_model.load_state_dict(dst_state)
    logger.info(f"Copied {matched} parameters")
    return matched


def register_hooks(model, outputs_dict, prefix=""):
    """Register forward hooks to capture intermediate outputs."""
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in outputs_dict:
                outputs_dict[name] = {'inputs': [], 'outputs': []}
            
            # Store input
            if isinstance(input, tuple):
                inp_to_store = []
                for i, inp in enumerate(input):
                    if isinstance(inp, torch.Tensor):
                        inp_to_store.append(inp.detach().cpu().clone())
                    else:
                        inp_to_store.append(inp)
                outputs_dict[name]['inputs'].append(tuple(inp_to_store))
            elif isinstance(input, torch.Tensor):
                outputs_dict[name]['inputs'].append(input.detach().cpu().clone())
            
            # Store output
            if isinstance(output, tuple):
                out_to_store = []
                for o in output:
                    if isinstance(o, torch.Tensor):
                        out_to_store.append(o.detach().cpu().clone())
                    else:
                        out_to_store.append(o)
                outputs_dict[name]['outputs'].append(tuple(out_to_store))
            elif isinstance(output, torch.Tensor):
                outputs_dict[name]['outputs'].append(output.detach().cpu().clone())
            
        return hook
    
    # Register hooks for key modules
    key_modules = [
        'preprocessor',
        'quantizer', 
        'mask_processor',
        'encoder',
        'encoder.pre_encode',
        'encoder.pos_enc',
        'decoder',
        'loss',
    ]
    
    for name in key_modules:
        try:
            module = model
            for part in name.split('.'):
                module = getattr(module, part)
            full_name = f"{prefix}{name}" if prefix else name
            h = module.register_forward_hook(make_hook(full_name))
            hooks.append(h)
        except AttributeError:
            pass  # Module doesn't exist
    
    return hooks


def compare_tensors(name: str, t1, t2, rtol=1e-4, atol=1e-5):
    """Compare two tensors and return comparison result."""
    if t1 is None or t2 is None:
        return {'name': name, 'status': 'SKIP', 'reason': 'None value'}
    
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        return {'name': name, 'status': 'SKIP', 'reason': 'Not tensors'}
    
    if t1.shape != t2.shape:
        return {
            'name': name, 
            'status': 'FAIL', 
            'reason': f'Shape mismatch: {t1.shape} vs {t2.shape}'
        }
    
    max_diff = (t1.float() - t2.float()).abs().max().item()
    mean_diff = (t1.float() - t2.float()).abs().mean().item()
    is_close = torch.allclose(t1.float(), t2.float(), rtol=rtol, atol=atol)
    
    return {
        'name': name,
        'status': 'PASS' if is_close else 'FAIL',
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'shape': list(t1.shape),
    }


def create_dummy_batch(batch_size=2, audio_len=16000, device='cpu'):
    """Create dummy batch for testing."""
    set_seed(42)
    audio = torch.randn(batch_size, audio_len, device=device)
    audio_lengths = torch.tensor([audio_len, audio_len - 1000], dtype=torch.int32, device=device)
    noise = torch.randn(batch_size, audio_len, device=device)
    noise_lengths = torch.tensor([audio_len, audio_len - 1000], dtype=torch.int32, device=device)
    noisy_audio = audio + 0.1 * noise
    noisy_audio_lengths = audio_lengths.clone()
    
    return AudioNoiseBatch(
        audio=audio,
        audio_len=audio_lengths,
        noise=noise,
        noise_len=noise_lengths,
        noisy_audio=noisy_audio,
        noisy_audio_len=noisy_audio_lengths,
    )


def run_comparison(config_path: str = None, output_dir: str = None, device: str = 'cpu'):
    """Main comparison function."""
    print("=" * 80)
    print("Comparing SimplifiedSSLModel with EncDecDenoiseMaskedTokenPredModel")
    print("=" * 80)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(project_root) / "comparison_outputs"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    if config_path is None:
        config_path = os.path.join(project_root, "config", "nest_fast-conformer.yaml")
    
    print(f"\n1. Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Disable data loader setup for faster initialization
    with OmegaConf.open_dict(cfg):
        if 'train_ds' in cfg.model:
            cfg.model.train_ds.defer_setup = True
        if 'validation_ds' in cfg.model:
            cfg.model.validation_ds.defer_setup = True
    
    # Create models
    print("\n2. Creating original model...")
    set_seed(42)
    original_model = EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=None)
    original_model.to(device)
    original_model.eval()
    print(f"   Original model params: {sum(p.numel() for p in original_model.parameters()):,}")
    
    print("\n3. Creating simplified model...")
    set_seed(42)
    simplified_model = SimplifiedSSLModel(cfg=cfg.model, trainer=None)
    simplified_model.to(device)
    simplified_model.eval()
    print(f"   Simplified model params: {sum(p.numel() for p in simplified_model.parameters()):,}")
    
    # Copy weights
    print("\n4. Copying weights from original to simplified...")
    matched = copy_weights(original_model, simplified_model)
    
    # Verify weights
    print("\n5. Verifying weights are identical...")
    weights_match = True
    for (name1, p1), (name2, p2) in zip(
        original_model.named_parameters(), simplified_model.named_parameters()
    ):
        if not torch.equal(p1.data, p2.data):
            print(f"   [FAIL] Weights differ: {name1}")
            weights_match = False
    
    if weights_match:
        print("   [PASS] All weights match exactly")
    
    # Create dummy batch
    print("\n6. Creating test batch...")
    batch = create_dummy_batch(batch_size=2, audio_len=16000, device=device)
    print(f"   Audio shape: {batch.audio.shape}")
    print(f"   Audio lengths: {batch.audio_len}")
    
    # Register hooks
    print("\n7. Registering forward hooks...")
    orig_outputs = {}
    simp_outputs = {}
    orig_hooks = register_hooks(original_model, orig_outputs, prefix="orig.")
    simp_hooks = register_hooks(simplified_model, simp_outputs, prefix="simp.")
    
    # Run forward pass
    print("\n8. Running forward pass on original model...")
    set_seed(42)
    with torch.no_grad():
        orig_result = original_model.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noise_signal=batch.noise,
            noise_signal_length=batch.noise_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=True,
        )
    
    print("   Running forward pass on simplified model...")
    set_seed(42)
    with torch.no_grad():
        simp_result = simplified_model.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noise_signal=batch.noise,
            noise_signal_length=batch.noise_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=True,
        )
    
    # Remove hooks
    for h in orig_hooks + simp_hooks:
        h.remove()
    
    # Compare final outputs
    print("\n9. Comparing final outputs...")
    results = []
    
    log_probs_orig, encoded_len_orig, masks_orig, tokens_orig = orig_result
    log_probs_simp, encoded_len_simp, masks_simp, tokens_simp = simp_result
    
    results.append(compare_tensors("log_probs", log_probs_orig.cpu(), log_probs_simp.cpu()))
    results.append(compare_tensors("encoded_len", encoded_len_orig.cpu().float(), encoded_len_simp.cpu().float()))
    results.append(compare_tensors("masks", masks_orig.cpu(), masks_simp.cpu()))
    results.append(compare_tensors("tokens", tokens_orig.cpu().float(), tokens_simp.cpu().float()))
    
    # Compare intermediate outputs
    print("\n10. Comparing intermediate outputs...")
    for module_name in ['preprocessor', 'quantizer', 'mask_processor', 'encoder', 'decoder']:
        orig_key = f"orig.{module_name}"
        simp_key = f"simp.{module_name}"
        
        if orig_key in orig_outputs and simp_key in simp_outputs:
            orig_out = orig_outputs[orig_key]['outputs']
            simp_out = simp_outputs[simp_key]['outputs']
            
            if orig_out and simp_out:
                # Get first output
                o1 = orig_out[0]
                o2 = simp_out[0]
                
                if isinstance(o1, tuple) and isinstance(o2, tuple):
                    for i, (t1, t2) in enumerate(zip(o1, o2)):
                        if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                            results.append(compare_tensors(f"{module_name}.output[{i}]", t1, t2))
                elif isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                    results.append(compare_tensors(f"{module_name}.output", o1, o2))
    
    # Run training step comparison
    print("\n11. Comparing training step...")
    set_seed(42)
    original_model.train()
    orig_train = original_model.training_step(batch, batch_idx=0)
    
    set_seed(42)
    simplified_model.train()
    simp_train = simplified_model.training_step(batch, batch_idx=0)
    
    results.append(compare_tensors("training_loss", 
                                   orig_train['loss'].cpu(), 
                                   simp_train['loss'].cpu()))
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    all_pass = True
    for r in results:
        status = r['status']
        name = r['name']
        
        if status == 'PASS':
            print(f"[PASS] {name}: max_diff={r.get('max_diff', 'N/A'):.2e}, shape={r.get('shape', 'N/A')}")
        elif status == 'FAIL':
            print(f"[FAIL] {name}: {r.get('reason', '')} max_diff={r.get('max_diff', 'N/A'):.2e}")
            all_pass = False
        else:
            print(f"[SKIP] {name}: {r.get('reason', '')}")
    
    # Save results
    results_path = output_dir / "comparison_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'orig_outputs': orig_outputs,
            'simp_outputs': simp_outputs,
            'orig_final': {
                'log_probs': log_probs_orig.cpu(),
                'encoded_len': encoded_len_orig.cpu(),
                'masks': masks_orig.cpu(),
                'tokens': tokens_orig.cpu(),
            },
            'simp_final': {
                'log_probs': log_probs_simp.cpu(),
                'encoded_len': encoded_len_simp.cpu(),
                'masks': masks_simp.cpu(),
                'tokens': tokens_simp.cpu(),
            },
            'orig_train_loss': orig_train['loss'].cpu(),
            'simp_train_loss': simp_train['loss'].cpu(),
        }, f)
    print(f"\nResults saved to: {results_path}")
    
    # Summary
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ SUCCESS: SimplifiedSSLModel is fully aligned with original!")
    else:
        print("✗ FAILURE: Some outputs do not match. Check details above.")
    print("=" * 80)
    
    return all_pass, results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare SimplifiedSSLModel with original")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    run_comparison(config_path=args.config, output_dir=args.output_dir, device=device)

