#!/usr/bin/env python3
"""
Compare intermediate layer outputs between NeMo and nest_ssl_project implementations.

This script:
1. Sets the same random seed for both models
2. Loads the same checkpoint or initializes with same weights
3. Uses the same input data
4. Captures intermediate outputs from each layer using forward hooks
5. Compares outputs layer by layer

Usage:
    python tools/compare_intermediate_outputs.py --ckpt_path <checkpoint_path> --nemo_ckpt_path <nemo_checkpoint_path>
    python tools/compare_intermediate_outputs.py --ckpt_path checkpoint.nemo --seed 42 --batch_size 2
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from typing import Dict, List, Tuple, Any

# Add project root to path
project_dir = Path(__file__).parent.parent  # nest_ssl_project directory
sys.path.insert(0, str(project_dir))

from models.ssl_models import EncDecDenoiseMaskedTokenPredModel
from utils.logging import get_logger

logger = get_logger(__name__)


class OutputHook:
    """Hook to capture intermediate outputs from model layers."""
    
    def __init__(self, name: str):
        self.name = name
        self.outputs = []
        self.inputs = []
        self.hook_handle = None
    
    def forward_hook(self, module, input, output):
        """Forward hook to capture module output."""
        # Store both input and output for comparison
        if isinstance(input, tuple):
            self.inputs.append([x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input])
        else:
            self.inputs.append(input.detach().clone() if isinstance(input, torch.Tensor) else input)
        
        if isinstance(output, tuple):
            self.outputs.append([x.detach().clone() if isinstance(x, torch.Tensor) else x for x in output])
        else:
            self.outputs.append(output.detach().clone() if isinstance(output, torch.Tensor) else output)
    
    def register(self, module):
        """Register hook on module."""
        self.hook_handle = module.register_forward_hook(self.forward_hook)
    
    def remove(self):
        """Remove hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def clear(self):
        """Clear captured outputs."""
        self.outputs = []
        self.inputs = []


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def register_hooks(model: nn.Module, hook_dict: Dict[str, OutputHook], prefix: str = ""):
    """Register hooks on all named modules."""
    for name, module in model.named_modules():
        if name == "":
            continue
        
        full_name = f"{prefix}.{name}" if prefix else name
        if full_name not in hook_dict:
            hook_dict[full_name] = OutputHook(full_name)
        hook_dict[full_name].register(module)


def create_test_input(batch_size: int = 2, seq_len: int = 16000, sample_rate: int = 16000, seed: int = 42):
    """Create test input audio signal."""
    set_seed(seed)
    
    # Create random audio signal (simulating real audio)
    audio_signal = torch.randn(batch_size, seq_len)
    
    # Create lengths (all sequences have same length for simplicity)
    audio_length = torch.full((batch_size,), seq_len, dtype=torch.long)
    
    return audio_signal, audio_length


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, name: str, atol: float = 1e-5, rtol: float = 1e-5) -> Dict[str, Any]:
    """Compare two tensors and return comparison results."""
    if tensor1.shape != tensor2.shape:
        return {
            'match': False,
            'reason': f'Shape mismatch: {tensor1.shape} vs {tensor2.shape}',
            'max_diff': None,
            'mean_diff': None,
            'max_abs_diff': None,
        }
    
    # Compute differences
    diff = tensor1 - tensor2
    max_diff = diff.max().item()
    min_diff = diff.min().item()
    mean_diff = diff.mean().item()
    max_abs_diff = diff.abs().max().item()
    
    # Check if tensors are close
    is_close = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
    
    return {
        'match': is_close,
        'reason': 'Match' if is_close else f'Max abs diff: {max_abs_diff:.2e}',
        'max_diff': max_diff,
        'min_diff': min_diff,
        'mean_diff': mean_diff,
        'max_abs_diff': max_abs_diff,
        'shape': tensor1.shape,
    }


def compare_outputs(output1: Any, output2: Any, name: str, atol: float = 1e-5, rtol: float = 1e-5) -> Dict[str, Any]:
    """Compare two outputs (can be tensor or tuple of tensors)."""
    if isinstance(output1, tuple) and isinstance(output2, tuple):
        results = []
        for i, (o1, o2) in enumerate(zip(output1, output2)):
            if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                result = compare_tensors(o1, o2, f"{name}[{i}]", atol, rtol)
                results.append(result)
            else:
                results.append({
                    'match': o1 == o2,
                    'reason': f'Non-tensor comparison: {type(o1)} vs {type(o2)}',
                })
        
        # Overall match if all elements match
        all_match = all(r.get('match', False) for r in results)
        return {
            'match': all_match,
            'results': results,
            'name': name,
        }
    elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
        return compare_tensors(output1, output2, name, atol, rtol)
    else:
        return {
            'match': output1 == output2,
            'reason': f'Type mismatch: {type(output1)} vs {type(output2)}',
        }


def load_local_model(ckpt_path: str, config=None):
    """Load local model from checkpoint."""
    if config is None:
        config_path = project_dir / 'config' / 'nest_fast-conformer.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg = OmegaConf.load(config_path)
        config = cfg.model
    
    if ckpt_path.endswith('.nemo'):
        model = EncDecDenoiseMaskedTokenPredModel.restore_from(ckpt_path)
    else:
        model = EncDecDenoiseMaskedTokenPredModel.load_from_checkpoint(
            ckpt_path,
            cfg=config,
            strict=False
        )
    
    model.eval()
    return model


def load_nemo_model(ckpt_path: str):
    """Load NeMo model from checkpoint."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        raise ImportError("NeMo is not available. Please install NeMo to compare with NeMo implementation.")
    
    if ckpt_path.endswith('.nemo'):
        try:
            model = nemo_asr.models.EncDecDenoiseMaskedTokenPredModel.restore_from(ckpt_path)
        except:
            model = nemo_asr.models.SpeechEncDecSelfSupervisedModel.restore_from(ckpt_path)
    else:
        model = nemo_asr.models.EncDecDenoiseMaskedTokenPredModel.load_from_checkpoint(ckpt_path)
    
    model.eval()
    return model


def compare_models(
    local_model: nn.Module,
    nemo_model: nn.Module,
    audio_signal: torch.Tensor,
    audio_length: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    """Compare intermediate outputs between two models."""
    
    # Register hooks on both models
    local_hooks = {}
    nemo_hooks = {}
    
    print("Registering hooks on local model...")
    register_hooks(local_model, local_hooks, prefix="local")
    
    print("Registering hooks on NeMo model...")
    register_hooks(nemo_model, nemo_hooks, prefix="nemo")
    
    # Forward pass on local model
    print("\nRunning forward pass on local model...")
    with torch.no_grad():
        local_output = local_model.forward(
            input_signal=audio_signal,
            input_signal_length=audio_length,
            apply_mask=False
        )
    
    # Forward pass on NeMo model
    print("Running forward pass on NeMo model...")
    with torch.no_grad():
        nemo_output = nemo_model.forward(
            input_signal=audio_signal,
            input_signal_length=audio_length,
            apply_mask=False
        )
    
    # Compare final outputs
    print("\n" + "="*80)
    print("Comparing Final Outputs")
    print("="*80)
    final_comparison = compare_outputs(local_output, nemo_output, "final_output", atol, rtol)
    print(f"Final output match: {final_comparison.get('match', False)}")
    if not final_comparison.get('match', False):
        print(f"  Reason: {final_comparison.get('reason', 'Unknown')}")
    
    # Compare intermediate outputs
    print("\n" + "="*80)
    print("Comparing Intermediate Layer Outputs")
    print("="*80)
    
    # Find common layers
    local_layer_names = set(local_hooks.keys())
    nemo_layer_names = set(nemo_hooks.keys())
    
    # Normalize layer names (remove prefix for comparison)
    local_normalized = {name.replace("local.", ""): name for name in local_layer_names}
    nemo_normalized = {name.replace("nemo.", ""): name for name in nemo_layer_names}
    
    common_layers = set(local_normalized.keys()) & set(nemo_normalized.keys())
    only_local = set(local_normalized.keys()) - set(nemo_normalized.keys())
    only_nemo = set(nemo_normalized.keys()) - set(local_normalized.keys())
    
    print(f"\nLayer statistics:")
    print(f"  Common layers: {len(common_layers)}")
    print(f"  Only in local: {len(only_local)}")
    print(f"  Only in NeMo: {len(only_nemo)}")
    
    # Compare common layers
    comparison_results = []
    matched_layers = 0
    mismatched_layers = 0
    
    print(f"\n{'Layer Name':<60} {'Match':<10} {'Max Abs Diff':<15} {'Shape':<20}")
    print("-" * 105)
    
    for layer_name in sorted(common_layers):
        local_hook = local_hooks[local_normalized[layer_name]]
        nemo_hook = nemo_hooks[nemo_normalized[layer_name]]
        
        # Get last output (from forward pass)
        if len(local_hook.outputs) == 0 or len(nemo_hook.outputs) == 0:
            continue
        
        local_out = local_hook.outputs[-1]
        nemo_out = nemo_hook.outputs[-1]
        
        result = compare_outputs(local_out, nemo_out, layer_name, atol, rtol)
        
        if isinstance(result, dict) and 'results' in result:
            # Tuple output
            all_match = all(r.get('match', False) for r in result['results'])
            max_diff = max(r.get('max_abs_diff', 0) for r in result['results'] if 'max_abs_diff' in r)
            shape_str = str(result['results'][0].get('shape', 'N/A'))
        else:
            all_match = result.get('match', False)
            max_diff = result.get('max_abs_diff', 0)
            shape_str = str(result.get('shape', 'N/A'))
        
        comparison_results.append({
            'layer': layer_name,
            'match': all_match,
            'result': result,
        })
        
        if all_match:
            matched_layers += 1
        else:
            mismatched_layers += 1
        
        status = "✓" if all_match else "✗"
        max_diff_str = f"{max_diff:.2e}" if max_diff > 0 else "0.00e+00"
        print(f"{layer_name:<60} {status:<10} {max_diff_str:<15} {shape_str:<20}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total common layers compared: {len(comparison_results)}")
    print(f"Matched layers: {matched_layers}")
    print(f"Mismatched layers: {mismatched_layers}")
    print(f"Match rate: {matched_layers / len(comparison_results) * 100:.2f}%")
    
    # Show details of mismatched layers
    if mismatched_layers > 0:
        print("\n" + "="*80)
        print("Mismatched Layers Details")
        print("="*80)
        for comp in comparison_results:
            if not comp['match']:
                print(f"\nLayer: {comp['layer']}")
                result = comp['result']
                if isinstance(result, dict) and 'results' in result:
                    for i, r in enumerate(result['results']):
                        if not r.get('match', False):
                            print(f"  Output[{i}]: {r.get('reason', 'Unknown')}")
                            if 'max_abs_diff' in r:
                                print(f"    Max abs diff: {r['max_abs_diff']:.2e}")
                else:
                    print(f"  {result.get('reason', 'Unknown')}")
                    if 'max_abs_diff' in result:
                        print(f"  Max abs diff: {result['max_abs_diff']:.2e}")
    
    # Cleanup hooks
    for hook in local_hooks.values():
        hook.remove()
    for hook in nemo_hooks.values():
        hook.remove()
    
    return comparison_results, final_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare intermediate outputs between NeMo and nest_ssl_project"
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        required=True,
        help='Path to local checkpoint file (.nemo or .ckpt)'
    )
    parser.add_argument(
        '--nemo_ckpt_path',
        type=str,
        required=True,
        help='Path to NeMo checkpoint file (.nemo or .ckpt)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size for test input (default: 2)'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=16000,
        help='Sequence length for test input (default: 16000)'
    )
    parser.add_argument(
        '--atol',
        type=float,
        default=1e-5,
        help='Absolute tolerance for comparison (default: 1e-5)'
    )
    parser.add_argument(
        '--rtol',
        type=float,
        default=1e-5,
        help='Relative tolerance for comparison (default: 1e-5)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model config file (optional)'
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("="*80)
    print("Intermediate Output Comparison")
    print("="*80)
    print(f"Local checkpoint: {args.ckpt_path}")
    print(f"NeMo checkpoint: {args.nemo_ckpt_path}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    print("="*80)
    
    # Load config if provided
    config = None
    if args.config:
        config = OmegaConf.load(args.config)
    
    # Load models
    print("\nLoading local model...")
    local_model = load_local_model(args.ckpt_path, config)
    
    print("Loading NeMo model...")
    nemo_model = load_nemo_model(args.nemo_ckpt_path)
    
    # Create test input
    print(f"\nCreating test input (seed={args.seed})...")
    audio_signal, audio_length = create_test_input(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed
    )
    
    print(f"Input shape: {audio_signal.shape}")
    print(f"Input length: {audio_length}")
    
    # Compare models
    comparison_results, final_comparison = compare_models(
        local_model,
        nemo_model,
        audio_signal,
        audio_length,
        atol=args.atol,
        rtol=args.rtol,
    )
    
    # Exit code
    if final_comparison.get('match', False):
        print("\n✓ Final outputs match!")
        sys.exit(0)
    else:
        print("\n✗ Final outputs do not match!")
        sys.exit(1)


if __name__ == '__main__':
    main()

