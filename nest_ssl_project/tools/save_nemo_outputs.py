#!/usr/bin/env python3
"""
Save NeMo model forward and backward intermediate outputs.

This script runs a NeMo model with a specific seed, captures all intermediate
outputs (forward and backward), and saves them to a file for later comparison.

Usage:
    python tools/save_nemo_outputs.py --ckpt_path <nemo_checkpoint> --output_dir <output_dir> --seed 42
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

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    raise ImportError("NeMo is required. Please run this script in NeMo environment.")


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
        # Store inputs
        if isinstance(input, tuple):
            self.forward_inputs.append([
                x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x 
                for x in input
            ])
        else:
            self.forward_inputs.append(
                input.detach().cpu().clone() if isinstance(input, torch.Tensor) else input
            )
        
        # Store outputs
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
        # Store input gradients (gradients w.r.t. inputs)
        if isinstance(grad_input, tuple):
            self.backward_input_grads.append([
                g.detach().cpu().clone() if isinstance(g, torch.Tensor) and g is not None else None
                for g in grad_input
            ])
        else:
            self.backward_input_grads.append(
                grad_input.detach().cpu().clone() if isinstance(grad_input, torch.Tensor) and grad_input is not None else None
            )
        
        # Store output gradients (gradients w.r.t. outputs)
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


def create_test_input(batch_size: int = 2, seq_len: int = 16000, seed: int = 42):
    """Create test input audio signal."""
    set_seed(seed)
    
    # Create random audio signal
    audio_signal = torch.randn(batch_size, seq_len)
    
    # Create lengths
    audio_length = torch.full((batch_size,), seq_len, dtype=torch.long)
    
    return audio_signal, audio_length


def register_hooks(model: nn.Module, hook_dict: Dict[str, ForwardBackwardHook], prefix: str = ""):
    """Register hooks on all named modules."""
    for name, module in model.named_modules():
        if name == "":
            continue
        
        full_name = f"{prefix}.{name}" if prefix else name
        if full_name not in hook_dict:
            hook_dict[full_name] = ForwardBackwardHook(full_name)
        hook_dict[full_name].register(module)


def save_parameter_gradients(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Save parameter gradients."""
    param_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_grads[name] = param.grad.detach().cpu().clone()
        else:
            param_grads[name] = None
    return param_grads


def load_nemo_model(ckpt_path: str):
    """Load NeMo model from checkpoint."""
    if ckpt_path.endswith('.nemo'):
        try:
            model = nemo_asr.models.EncDecDenoiseMaskedTokenPredModel.restore_from(ckpt_path)
        except:
            model = nemo_asr.models.SpeechEncDecSelfSupervisedModel.restore_from(ckpt_path)
    else:
        model = nemo_asr.models.EncDecDenoiseMaskedTokenPredModel.load_from_checkpoint(ckpt_path)
    
    model.train()  # Set to training mode to enable gradients
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Save NeMo model forward and backward outputs"
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        required=True,
        help='Path to NeMo checkpoint file (.nemo or .ckpt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory to save captured outputs'
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
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("="*80)
    print("Save NeMo Model Outputs")
    print("="*80)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading NeMo model...")
    model = load_nemo_model(args.ckpt_path)
    
    # Create test input
    print(f"\nCreating test input (seed={args.seed})...")
    audio_signal, audio_length = create_test_input(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed
    )
    
    print(f"Input shape: {audio_signal.shape}")
    print(f"Input length: {audio_length}")
    
    # Save input data
    torch.save({
        'audio_signal': audio_signal,
        'audio_length': audio_length,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
    }, output_dir / 'input_data.pt')
    print(f"\nSaved input data to {output_dir / 'input_data.pt'}")
    
    # Register hooks
    print("\nRegistering hooks on all modules...")
    hooks = {}
    register_hooks(model, hooks)
    print(f"Registered hooks on {len(hooks)} modules")
    
    # Forward pass
    print("\nRunning forward pass...")
    output = model.forward(
        input_signal=audio_signal,
        input_signal_length=audio_length,
        apply_mask=False
    )
    
    # Save forward output
    if isinstance(output, tuple):
        forward_output = [o.detach().cpu().clone() if isinstance(o, torch.Tensor) else o for o in output]
    else:
        forward_output = output.detach().cpu().clone() if isinstance(output, torch.Tensor) else output
    
    torch.save(forward_output, output_dir / 'forward_output.pt')
    print(f"Saved forward output to {output_dir / 'forward_output.pt'}")
    
    # Compute loss (dummy loss for backward pass)
    if isinstance(output, tuple):
        # Use first tensor output for loss
        loss_tensor = output[0]
        if isinstance(loss_tensor, torch.Tensor):
            loss = loss_tensor.mean()
        else:
            loss = torch.tensor(0.0, requires_grad=True)
    else:
        if isinstance(output, torch.Tensor):
            loss = output.mean()
        else:
            loss = torch.tensor(0.0, requires_grad=True)
    
    # Backward pass
    print("\nRunning backward pass...")
    loss.backward()
    
    # Save parameter gradients
    print("Saving parameter gradients...")
    param_grads = save_parameter_gradients(model)
    torch.save(param_grads, output_dir / 'parameter_gradients.pt')
    print(f"Saved parameter gradients to {output_dir / 'parameter_gradients.pt'}")
    
    # Collect hook data
    print("\nCollecting hook data...")
    hook_data = {}
    for name, hook in hooks.items():
        hook_data[name] = hook.get_data()
    
    # Save hook data
    print(f"Saving hook data for {len(hook_data)} modules...")
    with open(output_dir / 'hook_data.pkl', 'wb') as f:
        pickle.dump(hook_data, f)
    print(f"Saved hook data to {output_dir / 'hook_data.pkl'}")
    
    # Save metadata
    metadata = {
        'seed': args.seed,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'ckpt_path': args.ckpt_path,
        'num_modules': len(hooks),
        'module_names': list(hooks.keys()),
    }
    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {output_dir / 'metadata.pkl'}")
    
    # Cleanup hooks
    for hook in hooks.values():
        hook.remove()
    
    print("\n" + "="*80)
    print("Successfully saved all outputs!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Files saved:")
    print(f"  - input_data.pt")
    print(f"  - forward_output.pt")
    print(f"  - parameter_gradients.pt")
    print(f"  - hook_data.pkl")
    print(f"  - metadata.pkl")


if __name__ == '__main__':
    main()

