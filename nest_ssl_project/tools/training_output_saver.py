#!/usr/bin/env python3
"""
Training Output Saver - Save model outputs during training for comparison.

This module provides utilities to save model outputs during training,
which can be used to compare NeMo and nest_ssl_project implementations.

Usage in training script:
    from tools.training_output_saver import TrainingOutputSaver
    
    saver = TrainingOutputSaver(output_dir="./saved_outputs", seed=42)
    saver.setup_hooks(model)
    
    # During training
    for batch in dataloader:
        output = model(batch)
        loss = compute_loss(output)
        loss.backward()
        saver.save_step(step=current_step)  # Save outputs for this step
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import OrderedDict


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
        """Backward hook to capture gradients (for register_full_backward_hook).
        
        NOTE: This hook is disabled by default to avoid inplace modification errors.
        Even capturing grad_input can cause issues when tensors are views that get
        modified inplace by subsequent operations (e.g., ReLU).
        We rely on parameter gradients captured via param.grad in save_step() instead.
        """
        # Do not capture anything to avoid inplace modification errors
        # Parameter gradients are captured in save_step() via param.grad
        return None
    
    def register(self, module):
        """Register hooks on module."""
        self.forward_hook_handle = module.register_forward_hook(self.forward_hook)
        # DISABLED: Do not register backward hook to avoid inplace modification errors
        # Even capturing grad_input can cause RuntimeError when tensors are views
        # that get modified inplace (e.g., ReLU inplace operations).
        # Parameter gradients are still captured in save_step() via param.grad
        self.backward_hook_handle = None
    
    def remove(self):
        """Remove hooks."""
        if self.forward_hook_handle is not None:
            self.forward_hook_handle.remove()
            self.forward_hook_handle = None
        if self.backward_hook_handle is not None:
            self.backward_hook_handle.remove()
            self.backward_hook_handle = None
    
    def get_data(self):
        """Get captured data (all forward/backward passes)."""
        return {
            'forward_inputs': self.forward_inputs[-1] if self.forward_inputs else None,
            'forward_outputs': self.forward_outputs[-1] if self.forward_outputs else None,
            'backward_input_grads': self.backward_input_grads[-1] if self.backward_input_grads else None,
            'backward_output_grads': self.backward_output_grads[-1] if self.backward_output_grads else None,
            'all_forward_inputs': self.forward_inputs if self.forward_inputs else [],
            'all_forward_outputs': self.forward_outputs if self.forward_outputs else [],
        }
    
    def clear(self):
        """Clear captured data."""
        self.forward_inputs = []
        self.forward_outputs = []
        self.backward_input_grads = []
        self.backward_output_grads = []


class TrainingOutputSaver:
    """Save training outputs for comparison."""
    
    def __init__(
        self,
        output_dir: str,
        seed: int,
        save_every_n_steps: int = 1,
        save_first_n_steps: int = 5,
    ):
        """
        Initialize output saver.
        
        Args:
            output_dir: Directory to save outputs
            seed: Random seed used for training
            save_every_n_steps: Save outputs every N steps (default: 1, save all)
            save_first_n_steps: Always save first N steps (default: 5)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.save_every_n_steps = save_every_n_steps
        self.save_first_n_steps = save_first_n_steps
        
        self.hooks = {}
        self.model = None
        self.saved_steps = []
        
        # Store metadata
        self.metadata = {
            'seed': seed,
            'save_every_n_steps': save_every_n_steps,
            'save_first_n_steps': save_first_n_steps,
        }
    
    def setup_hooks(self, model: nn.Module, prefix: str = ""):
        """Register hooks on all modules."""
        self.model = model
        self.hooks = {}
        
        for name, module in model.named_modules():
            if name == "":
                continue
            
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name not in self.hooks:
                self.hooks[full_name] = ForwardBackwardHook(full_name)
            self.hooks[full_name].register(module)
    
    def save_step(
        self,
        step: int,
        batch: Any,
        forward_output: Any,
        loss: torch.Tensor,
        save_batch: bool = True,
        save_weights: bool = False,
        force_save: bool = False,
    ):
        """
        Save outputs for a training step.
        
        Args:
            step: Training step number
            batch: Input batch
            forward_output: Model forward output
            loss: Loss tensor
            save_batch: Whether to save batch data (default: True)
            save_weights: Whether to save model weights (default: False, set True after optimizer.step())
            force_save: If True, skip the should_save check (default: False)
        """
        # Check if we should save this step (unless force_save is True)
        if not force_save:
            should_save = (
                step < self.save_first_n_steps or
                step % self.save_every_n_steps == 0
            )
            
            if not should_save:
                return
        
        step_dir = self.output_dir / f"step_{step}"
        step_dir.mkdir(exist_ok=True)
        
        # Save batch data
        if save_batch:
            batch_data = {}
            # Handle AudioNoiseBatch (dataclass) or similar custom batch types
            if hasattr(batch, '__dict__') or hasattr(type(batch), '__annotations__'):
                # Extract all tensor fields from batch object
                for attr in ['audio', 'audio_len', 'noise', 'noise_len', 'noisy_audio', 'noisy_audio_len', 'sample_id']:
                    if hasattr(batch, attr):
                        val = getattr(batch, attr)
                        if isinstance(val, torch.Tensor):
                            batch_data[attr] = val.detach().cpu().clone()
                        else:
                            batch_data[attr] = val
                # Also save all __dict__ attributes
                if hasattr(batch, '__dict__'):
                    for k, v in batch.__dict__.items():
                        if k not in batch_data:
                            if isinstance(v, torch.Tensor):
                                batch_data[k] = v.detach().cpu().clone()
                            else:
                                batch_data[k] = v
            elif isinstance(batch, (list, tuple)):
                batch_data = {
                    f'batch_{i}': x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x
                    for i, x in enumerate(batch)
                }
            elif isinstance(batch, dict):
                batch_data = {
                    k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            else:
                batch_data = {'batch': batch.detach().cpu().clone() if isinstance(batch, torch.Tensor) else batch}
            
            torch.save(batch_data, step_dir / 'batch.pt')
        
        # Save forward output
        if isinstance(forward_output, (list, tuple)):
            forward_output_data = [
                x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x
                for x in forward_output
            ]
        else:
            forward_output_data = forward_output.detach().cpu().clone() if isinstance(forward_output, torch.Tensor) else forward_output
        
        torch.save(forward_output_data, step_dir / 'forward_output.pt')
        
        # Save loss
        torch.save(loss.detach().cpu().clone(), step_dir / 'loss.pt')
        
        # Save parameter gradients (before optimizer step)
        param_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_grads[name] = param.grad.detach().cpu().clone()
            else:
                param_grads[name] = None
        
        torch.save(param_grads, step_dir / 'parameter_gradients.pt')
        
        # Save all layer forward outputs (from hooks)
        layer_outputs = {}
        for name, hook in self.hooks.items():
            hook_data = hook.get_data()
            # Only save forward outputs (not backward gradients to avoid inplace errors)
            layer_outputs[name] = {
                'forward_inputs': hook_data.get('forward_inputs'),
                'forward_outputs': hook_data.get('forward_outputs'),
                'all_forward_inputs': hook_data.get('all_forward_inputs', []),
                'all_forward_outputs': hook_data.get('all_forward_outputs', []),
            }
        
        with open(step_dir / 'layer_outputs.pkl', 'wb') as f:
            pickle.dump(layer_outputs, f)
        
        # Save model weights (after optimizer step)
        if save_weights:
            param_weights = {}
            for name, param in self.model.named_parameters():
                param_weights[name] = param.detach().cpu().clone()
            
            torch.save(param_weights, step_dir / 'parameter_weights.pt')
        
        # Save step metadata
        step_metadata = {
            'step': step,
            'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'num_modules': len(self.hooks),
            'saved_weights': save_weights,
        }
        with open(step_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(step_metadata, f)
        
        self.saved_steps.append(step)
        
        # Clear hooks for next step
        for hook in self.hooks.values():
            hook.clear()
    
    def save_model_structure(self, model: nn.Module):
        """Save model structure information."""
        structure = {
            'module_names': list(self.hooks.keys()),
            'num_modules': len(self.hooks),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        
        # Save parameter names and shapes
        param_info = {}
        for name, param in model.named_parameters():
            param_info[name] = {
                'shape': list(param.shape),
                'dtype': str(param.dtype),
                'requires_grad': param.requires_grad,
            }
        
        structure['parameters'] = param_info
        
        with open(self.output_dir / 'model_structure.pkl', 'wb') as f:
            pickle.dump(structure, f)

    def save_buffers(self, model: nn.Module):
        """Save all buffers of the module."""
        buffer_dir = self.output_dir / "buffers"
        buffer_dir.mkdir(exist_ok=True)
        
        buffers = {}
        buffer_count = 0
        for name, buf in model.named_buffers():
            buffers[name] = buf.detach().cpu().clone()
            buffer_count += 1
        
        buffer_file = buffer_dir / "buffers.pt"
        torch.save(buffers, buffer_file)
        
        # Log success
        print(f"[TrainingOutputSaver] Saved {buffer_count} buffers to {buffer_file}")
        
        # Also try to save specifically to layer_outputs structure if possible, or just leave it here.
        # For diagnose_featurizer.py, it expects them in layer_outputs.pkl under module name.
        # We can't easily inject into layer_outputs.pkl since that's per step.
        # But diagnose_featurizer.py can be updated to load buffers.pt.
    
    def finalize(self):
        """Finalize and save metadata."""
        self.metadata['saved_steps'] = self.saved_steps
        self.metadata['num_saved_steps'] = len(self.saved_steps)
        
        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"\nSaved outputs for {len(self.saved_steps)} steps to {self.output_dir}")
    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}


class TrainingOutputComparator:
    """Compare training outputs with saved NeMo outputs."""
    
    def __init__(
        self,
        saved_outputs_dir: str,
        comparison_output_dir: Optional[str] = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        """
        Initialize output comparator.
        
        Args:
            saved_outputs_dir: Directory containing saved NeMo outputs
            comparison_output_dir: Directory to save comparison results (optional)
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
        """
        self.saved_outputs_dir = Path(saved_outputs_dir)
        self.comparison_output_dir = Path(comparison_output_dir) if comparison_output_dir else None
        if self.comparison_output_dir:
            self.comparison_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.atol = atol
        self.rtol = rtol
        
        # Load saved metadata
        with open(self.saved_outputs_dir / 'metadata.pkl', 'rb') as f:
            self.saved_metadata = pickle.load(f)
        
        self.comparison_results = []
    
    def compare_step(
        self,
        step: int,
        batch: Any,
        forward_output: Any,
        loss: torch.Tensor,
        model: nn.Module,
        hooks: Dict[str, ForwardBackwardHook],
    ) -> Dict[str, Any]:
        """
        Compare outputs for a training step.
        
        Returns:
            Comparison results dictionary
        """
        step_dir = self.saved_outputs_dir / f"step_{step}"
        if not step_dir.exists():
            return {'error': f'Step {step} not found in saved outputs'}
        
        # Load saved data
        batch_saved = torch.load(step_dir / 'batch.pt')
        forward_output_saved = torch.load(step_dir / 'forward_output.pt')
        loss_saved = torch.load(step_dir / 'loss.pt')
        param_grads_saved = torch.load(step_dir / 'parameter_gradients.pt')
        
        with open(step_dir / 'hook_data.pkl', 'rb') as f:
            hook_data_saved = pickle.load(f)
        
        # Compare forward outputs
        forward_match = self._compare_tensors(forward_output, forward_output_saved, 'forward_output')
        
        # Compare loss
        loss_match = self._compare_tensors(loss, loss_saved, 'loss')
        
        # Compare parameter gradients
        param_grad_matches = {}
        param_grads_local = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grads_local[name] = param.grad.detach().cpu().clone()
            else:
                param_grads_local[name] = None
        
        common_params = set(param_grads_local.keys()) & set(param_grads_saved.keys())
        for param_name in common_params:
            grad_local = param_grads_local[param_name]
            grad_saved = param_grads_saved[param_name]
            param_grad_matches[param_name] = self._compare_tensors(grad_local, grad_saved, f'param_{param_name}')
        
        # Compare intermediate outputs
        hook_data_local = {}
        for name, hook in hooks.items():
            hook_data_local[name] = hook.get_data()
        
        # Normalize layer names
        local_normalized = {name.replace("local.", ""): name for name in hook_data_local.keys()}
        saved_normalized = {name.replace("nemo.", ""): name for name in hook_data_saved.keys()}
        
        common_layers = set(local_normalized.keys()) & set(saved_normalized.keys())
        layer_matches = {}
        for layer_name in common_layers:
            local_hook = hook_data_local[local_normalized[layer_name]]
            saved_hook = hook_data_saved[saved_normalized[layer_name]]
            
            forward_match = self._compare_outputs(
                local_hook['forward_outputs'],
                saved_hook['forward_outputs'],
                f'{layer_name}.forward_output'
            )
            backward_match = self._compare_outputs(
                local_hook['backward_output_grads'],
                saved_hook['backward_output_grads'],
                f'{layer_name}.backward_grad'
            )
            
            layer_matches[layer_name] = {
                'forward': forward_match,
                'backward': backward_match,
            }
        
        result = {
            'step': step,
            'forward_output_match': forward_match,
            'loss_match': loss_match,
            'parameter_gradients': param_grad_matches,
            'layer_outputs': layer_matches,
            'num_common_layers': len(common_layers),
            'num_common_params': len(common_params),
        }
        
        self.comparison_results.append(result)
        
        # Save comparison result if output dir specified
        if self.comparison_output_dir:
            with open(self.comparison_output_dir / f'comparison_step_{step}.pkl', 'wb') as f:
                pickle.dump(result, f)
        
        return result
    
    def _compare_tensors(self, tensor1: Any, tensor2: Any, name: str) -> Dict[str, Any]:
        """Compare two tensors."""
        if tensor1 is None or tensor2 is None:
            return {
                'match': tensor1 is None and tensor2 is None,
                'reason': 'One or both are None',
            }
        
        # Handle tuple/list
        if isinstance(tensor1, (list, tuple)) and isinstance(tensor2, (list, tuple)):
            results = []
            for i, (t1, t2) in enumerate(zip(tensor1, tensor2)):
                results.append(self._compare_tensors(t1, t2, f'{name}[{i}]'))
            return {
                'match': all(r.get('match', False) for r in results),
                'results': results,
            }
        
        if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
            return {
                'match': tensor1 == tensor2,
                'reason': f'Type mismatch: {type(tensor1)} vs {type(tensor2)}',
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
        
        is_close = torch.allclose(tensor1, tensor2, atol=self.atol, rtol=self.rtol)
        
        return {
            'match': is_close,
            'reason': 'Match' if is_close else f'Max abs diff: {max_abs_diff:.2e}',
            'max_abs_diff': max_abs_diff,
            'mean_diff': mean_diff,
            'shape': list(tensor1.shape),
        }
    
    def _compare_outputs(self, output1: Any, output2: Any, name: str) -> Dict[str, Any]:
        """Compare two outputs (can be tensor or tuple)."""
        return self._compare_tensors(output1, output2, name)
    
    def print_summary(self):
        """Print comparison summary."""
        if not self.comparison_results:
            print("No comparison results available.")
            return
        
        print("\n" + "="*80)
        print("Comparison Summary")
        print("="*80)
        
        total_steps = len(self.comparison_results)
        forward_matches = sum(1 for r in self.comparison_results if r.get('forward_output_match', {}).get('match', False))
        loss_matches = sum(1 for r in self.comparison_results if r.get('loss_match', {}).get('match', False))
        
        print(f"Total steps compared: {total_steps}")
        print(f"Forward output matches: {forward_matches}/{total_steps} ({forward_matches/total_steps*100:.2f}%)")
        print(f"Loss matches: {loss_matches}/{total_steps} ({loss_matches/total_steps*100:.2f}%)")
        
        # Parameter gradient statistics
        if self.comparison_results:
            first_result = self.comparison_results[0]
            param_grads = first_result.get('parameter_gradients', {})
            if param_grads:
                param_matches = sum(1 for v in param_grads.values() if v.get('match', False))
                print(f"Parameter gradients (first step): {param_matches}/{len(param_grads)} ({param_matches/len(param_grads)*100:.2f}%)")
        
        # Layer output statistics
        if self.comparison_results:
            first_result = self.comparison_results[0]
            layer_outputs = first_result.get('layer_outputs', {})
            if layer_outputs:
                forward_layer_matches = sum(1 for v in layer_outputs.values() if v.get('forward', {}).get('match', False))
                backward_layer_matches = sum(1 for v in layer_outputs.values() if v.get('backward', {}).get('match', False))
                print(f"Layer forward matches (first step): {forward_layer_matches}/{len(layer_outputs)} ({forward_layer_matches/len(layer_outputs)*100:.2f}%)")
                print(f"Layer backward matches (first step): {backward_layer_matches}/{len(layer_outputs)} ({backward_layer_matches/len(layer_outputs)*100:.2f}%)")

