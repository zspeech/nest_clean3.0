#!/usr/bin/env python3
"""
Compare nest_ssl_project model outputs with saved NeMo outputs.

This script runs a nest_ssl_project model with the same seed and input as NeMo,
captures intermediate outputs, and compares them with saved NeMo outputs.

Usage:
    python tools/compare_with_saved_outputs.py --ckpt_path <checkpoint> --saved_outputs_dir <saved_dir>
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pickle
from omegaconf import OmegaConf
from collections import OrderedDict
from typing import Dict, List, Tuple, Any

# Add project root to path
project_dir = Path(__file__).parent.parent  # nest_ssl_project directory
sys.path.insert(0, str(project_dir))

from models.ssl_models import EncDecDenoiseMaskedTokenPredModel
from utils.logging import get_logger

logger = get_logger(__name__)


class ForwardBackwardHook:
    """Hook to capture forward and backward outputs."""
    
    def __init__(self, name: str):
        self.name = name
        self.forward_inputs = []
        self.forward_outputs = []
        self.backward_input_grads = []
        self.backward_output_grads = []
        self.forward_hook_handle = None
        self.backward_hook_handle = None
    
    def forward_hook(self, module, input, output):
        """Forward hook to capture module input and output."""
        if isinstance(input, tuple):
            self.forward_inputs.append([
                x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x 
                for x in input
            ])
        else:
            self.forward_inputs.append(
                input.detach().cpu().clone() if isinstance(input, torch.Tensor) else input
            )
        
        if isinstance(output, tuple):
            self.forward_outputs.append([
                x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x 
                for x in output
            ])
        else:
            self.forward_outputs.append(
                output.detach().cpu().clone() if isinstance(output, torch.Tensor) else output
            )
    
    def backward_hook(self, module, grad_input, grad_output):
        """Backward hook to capture gradients."""
        if isinstance(grad_input, tuple):
            self.backward_input_grads.append([
                g.detach().cpu().clone() if isinstance(g, torch.Tensor) and g is not None else None
                for g in grad_input
            ])
        else:
            self.backward_input_grads.append(
                grad_input.detach().cpu().clone() if isinstance(grad_input, torch.Tensor) and grad_input is not None else None
            )
        
        if isinstance(grad_output, tuple):
            self.backward_output_grads.append([
                g.detach().cpu().clone() if isinstance(g, torch.Tensor) and g is not None else None
                for g in grad_output
            ])
        else:
            self.backward_output_grads.append(
                grad_output.detach().cpu().clone() if isinstance(grad_output, torch.Tensor) and grad_output is not None else None
            )
    
    def register(self, module):
        """Register hooks on module."""
        self.forward_hook_handle = module.register_forward_hook(self.forward_hook)
        self.backward_hook_handle = module.register_full_backward_hook(self.backward_hook)
    
    def remove(self):
        """Remove hooks."""
        if self.forward_hook_handle is not None:
            self.forward_hook_handle.remove()
            self.forward_hook_handle = None
        if self.backward_hook_handle is not None:
            self.backward_hook_handle.remove()
            self.backward_hook_handle = None
    
    def get_data(self):
        """Get captured data."""
        return {
            'forward_inputs': self.forward_inputs[-1] if self.forward_inputs else None,
            'forward_outputs': self.forward_outputs[-1] if self.forward_outputs else None,
            'backward_input_grads': self.backward_input_grads[-1] if self.backward_input_grads else None,
            'backward_output_grads': self.backward_output_grads[-1] if self.backward_output_grads else None,
        }


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def register_hooks(model: nn.Module, hook_dict: Dict[str, ForwardBackwardHook], prefix: str = ""):
    """Register hooks on all named modules."""
    for name, module in model.named_modules():
        if name == "":
            continue
        
        full_name = f"{prefix}.{name}" if prefix else name
        if full_name not in hook_dict:
            hook_dict[full_name] = ForwardBackwardHook(full_name)
        hook_dict[full_name].register(module)


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, name: str, atol: float = 1e-5, rtol: float = 1e-5) -> Dict[str, Any]:
    """Compare two tensors."""
    if tensor1 is None or tensor2 is None:
        return {
            'match': tensor1 is None and tensor2 is None,
            'reason': 'One or both tensors are None',
        }
    
    if tensor1.shape != tensor2.shape:
        return {
            'match': False,
            'reason': f'Shape mismatch: {tensor1.shape} vs {tensor2.shape}',
            'max_abs_diff': None,
        }
    
    diff = tensor1 - tensor2
    max_abs_diff = diff.abs().max().item()
    mean_diff = diff.mean().item()
    
    is_close = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
    
    return {
        'match': is_close,
        'reason': 'Match' if is_close else f'Max abs diff: {max_abs_diff:.2e}',
        'max_abs_diff': max_abs_diff,
        'mean_diff': mean_diff,
        'shape': tensor1.shape,
    }


def compare_outputs(output1: Any, output2: Any, name: str, atol: float = 1e-5, rtol: float = 1e-5) -> Dict[str, Any]:
    """Compare two outputs."""
    if isinstance(output1, tuple) and isinstance(output2, tuple):
        results = []
        for i, (o1, o2) in enumerate(zip(output1, output2)):
            if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                result = compare_tensors(o1, o2, f"{name}[{i}]", atol, rtol)
                results.append(result)
            else:
                results.append({
                    'match': o1 == o2,
                    'reason': f'Non-tensor: {type(o1)} vs {type(o2)}',
                })
        
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
    
    model.train()  # Set to training mode to enable gradients
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Compare nest_ssl_project outputs with saved NeMo outputs"
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        required=True,
        help='Path to local checkpoint file (.nemo or .ckpt)'
    )
    parser.add_argument(
        '--saved_outputs_dir',
        type=str,
        required=True,
        help='Directory containing saved NeMo outputs'
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
    
    saved_dir = Path(args.saved_outputs_dir)
    if not saved_dir.exists():
        raise FileNotFoundError(f"Saved outputs directory not found: {saved_dir}")
    
    # Load saved data
    print("="*80)
    print("Loading saved NeMo outputs...")
    print("="*80)
    
    input_data = torch.load(saved_dir / 'input_data.pt')
    forward_output_nemo = torch.load(saved_dir / 'forward_output.pt')
    param_grads_nemo = torch.load(saved_dir / 'parameter_gradients.pt')
    
    with open(saved_dir / 'hook_data.pkl', 'rb') as f:
        hook_data_nemo = pickle.load(f)
    
    with open(saved_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    seed = metadata['seed']
    print(f"Loaded data with seed: {seed}")
    print(f"Batch size: {metadata['batch_size']}")
    print(f"Sequence length: {metadata['seq_len']}")
    
    # Set same seed
    set_seed(seed)
    
    # Load input data
    audio_signal = input_data['audio_signal']
    audio_length = input_data['audio_length']
    
    print(f"\nInput shape: {audio_signal.shape}")
    print(f"Input length: {audio_length}")
    
    # Load config if provided
    config = None
    if args.config:
        config = OmegaConf.load(args.config)
    
    # Load local model
    print("\nLoading local model...")
    local_model = load_local_model(args.ckpt_path, config)
    
    # Register hooks
    print("\nRegistering hooks...")
    hooks = {}
    register_hooks(local_model, hooks)
    print(f"Registered hooks on {len(hooks)} modules")
    
    # Forward pass
    print("\nRunning forward pass...")
    forward_output_local = local_model.forward(
        input_signal=audio_signal,
        input_signal_length=audio_length,
        apply_mask=False
    )
    
    # Compare forward output
    print("\n" + "="*80)
    print("Comparing Final Forward Output")
    print("="*80)
    forward_comparison = compare_outputs(forward_output_local, forward_output_nemo, "forward_output", args.atol, args.rtol)
    print(f"Forward output match: {forward_comparison.get('match', False)}")
    if not forward_comparison.get('match', False):
        if 'results' in forward_comparison:
            for i, r in enumerate(forward_comparison['results']):
                print(f"  Output[{i}]: {r.get('reason', 'Unknown')}")
        else:
            print(f"  {forward_comparison.get('reason', 'Unknown')}")
    
    # Compute loss for backward pass
    if isinstance(forward_output_local, tuple):
        loss_tensor = forward_output_local[0]
        if isinstance(loss_tensor, torch.Tensor):
            loss = loss_tensor.mean()
        else:
            loss = torch.tensor(0.0, requires_grad=True)
    else:
        if isinstance(forward_output_local, torch.Tensor):
            loss = forward_output_local.mean()
        else:
            loss = torch.tensor(0.0, requires_grad=True)
    
    # Backward pass
    print("\nRunning backward pass...")
    loss.backward()
    
    # Save parameter gradients
    param_grads_local = {}
    for name, param in local_model.named_parameters():
        if param.grad is not None:
            param_grads_local[name] = param.grad.detach().cpu().clone()
        else:
            param_grads_local[name] = None
    
    # Compare parameter gradients
    print("\n" + "="*80)
    print("Comparing Parameter Gradients")
    print("="*80)
    
    common_params = set(param_grads_local.keys()) & set(param_grads_nemo.keys())
    print(f"Common parameters: {len(common_params)}")
    
    param_grad_matches = 0
    param_grad_mismatches = 0
    
    print(f"\n{'Parameter Name':<60} {'Match':<10} {'Max Abs Diff':<15}")
    print("-" * 85)
    
    for param_name in sorted(common_params):
        grad_local = param_grads_local[param_name]
        grad_nemo = param_grads_nemo[param_name]
        
        if grad_local is None or grad_nemo is None:
            match = grad_local is None and grad_nemo is None
            reason = 'Both None' if match else 'One is None'
        else:
            result = compare_tensors(grad_local, grad_nemo, param_name, args.atol, args.rtol)
            match = result.get('match', False)
            reason = result.get('reason', 'Unknown')
            max_diff = result.get('max_abs_diff', 0)
        
        if match:
            param_grad_matches += 1
        else:
            param_grad_mismatches += 1
        
        status = "✓" if match else "✗"
        max_diff_str = f"{max_diff:.2e}" if 'max_diff' in locals() and max_diff > 0 else "N/A"
        print(f"{param_name:<60} {status:<10} {max_diff_str:<15}")
    
    # Compare intermediate outputs
    print("\n" + "="*80)
    print("Comparing Intermediate Layer Outputs")
    print("="*80)
    
    # Get local hook data
    hook_data_local = {}
    for name, hook in hooks.items():
        hook_data_local[name] = hook.get_data()
    
    # Normalize layer names for comparison
    local_normalized = {name.replace("local.", ""): name for name in hook_data_local.keys()}
    nemo_normalized = {name.replace("nemo.", ""): name for name in hook_data_nemo.keys()}
    
    common_layers = set(local_normalized.keys()) & set(nemo_normalized.keys())
    
    print(f"\nCommon layers: {len(common_layers)}")
    
    # Compare forward outputs
    print(f"\n{'Layer Name':<60} {'Forward Match':<15} {'Backward Match':<15} {'Max Diff':<15}")
    print("-" * 105)
    
    forward_matches = 0
    forward_mismatches = 0
    backward_matches = 0
    backward_mismatches = 0
    
    for layer_name in sorted(common_layers):
        local_hook = hook_data_local[local_normalized[layer_name]]
        nemo_hook = hook_data_nemo[nemo_normalized[layer_name]]
        
        # Compare forward outputs
        forward_match = False
        forward_max_diff = 0
        if local_hook['forward_outputs'] is not None and nemo_hook['forward_outputs'] is not None:
            forward_result = compare_outputs(
                local_hook['forward_outputs'],
                nemo_hook['forward_outputs'],
                f"{layer_name}.forward_output",
                args.atol,
                args.rtol
            )
            forward_match = forward_result.get('match', False)
            if 'results' in forward_result:
                forward_max_diff = max(r.get('max_abs_diff', 0) for r in forward_result['results'] if 'max_abs_diff' in r)
            else:
                forward_max_diff = forward_result.get('max_abs_diff', 0)
        
        # Compare backward output grads
        backward_match = False
        backward_max_diff = 0
        if local_hook['backward_output_grads'] is not None and nemo_hook['backward_output_grads'] is not None:
            backward_result = compare_outputs(
                local_hook['backward_output_grads'],
                nemo_hook['backward_output_grads'],
                f"{layer_name}.backward_output_grad",
                args.atol,
                args.rtol
            )
            backward_match = backward_result.get('match', False)
            if 'results' in backward_result:
                backward_max_diff = max(r.get('max_abs_diff', 0) for r in backward_result['results'] if 'max_abs_diff' in r)
            else:
                backward_max_diff = backward_result.get('max_abs_diff', 0)
        
        if forward_match:
            forward_matches += 1
        else:
            forward_mismatches += 1
        
        if backward_match:
            backward_matches += 1
        else:
            backward_mismatches += 1
        
        forward_status = "✓" if forward_match else "✗"
        backward_status = "✓" if backward_match else "✗"
        max_diff_str = f"{max(forward_max_diff, backward_max_diff):.2e}" if max(forward_max_diff, backward_max_diff) > 0 else "0.00e+00"
        
        print(f"{layer_name:<60} {forward_status:<15} {backward_status:<15} {max_diff_str:<15}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Forward output match: {forward_comparison.get('match', False)}")
    print(f"\nParameter gradients:")
    print(f"  Matched: {param_grad_matches}")
    print(f"  Mismatched: {param_grad_mismatches}")
    print(f"  Match rate: {param_grad_matches / len(common_params) * 100:.2f}%")
    print(f"\nIntermediate layers:")
    print(f"  Forward matches: {forward_matches}")
    print(f"  Forward mismatches: {forward_mismatches}")
    print(f"  Backward matches: {backward_matches}")
    print(f"  Backward mismatches: {backward_mismatches}")
    print(f"  Forward match rate: {forward_matches / len(common_layers) * 100:.2f}%")
    print(f"  Backward match rate: {backward_matches / len(common_layers) * 100:.2f}%")
    
    # Cleanup
    for hook in hooks.values():
        hook.remove()
    
    # Exit code
    all_match = (
        forward_comparison.get('match', False) and
        param_grad_mismatches == 0 and
        forward_mismatches == 0 and
        backward_mismatches == 0
    )
    
    if all_match:
        print("\n✓ All outputs match!")
        sys.exit(0)
    else:
        print("\n✗ Some outputs do not match!")
        sys.exit(1)


if __name__ == '__main__':
    main()

