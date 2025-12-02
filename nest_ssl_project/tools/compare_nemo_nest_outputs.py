#!/usr/bin/env python3
"""
Compare outputs between NeMo and nest_ssl_project implementations.

This script compares:
1. Model structure (architecture alignment)
2. All layer outputs (forward pass)
3. Forward output
4. Loss calculation
5. Gradients

Usage:
    python tools/compare_nemo_nest_outputs.py \
        --nemo_output_dir ./saved_nemo_outputs \
        --nest_output_dir ./saved_nest_outputs \
        --step 0 \
        --atol 1e-5 \
        --rtol 1e-5
"""

import argparse
import sys
from pathlib import Path
import torch
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict


def compare_tensors(
    tensor1: Any,
    tensor2: Any,
    name: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Dict[str, Any]:
    """Compare two tensors."""
    if tensor1 is None or tensor2 is None:
        return {
            'match': tensor1 is None and tensor2 is None,
            'reason': 'One or both are None',
            'max_abs_diff': None,
            'mean_diff': None,
            'shape': None,
        }
    
    # Handle tuple/list
    if isinstance(tensor1, (list, tuple)) and isinstance(tensor2, (list, tuple)):
        if len(tensor1) != len(tensor2):
            return {
                'match': False,
                'reason': f'Length mismatch: {len(tensor1)} vs {len(tensor2)}',
                'max_abs_diff': None,
                'mean_diff': None,
                'shape': None,
            }
        
        results = []
        all_match = True
        max_diff = 0.0
        for i, (t1, t2) in enumerate(zip(tensor1, tensor2)):
            result = compare_tensors(t1, t2, f'{name}[{i}]', atol, rtol)
            results.append(result)
            if not result.get('match', False):
                all_match = False
            if result.get('max_abs_diff') is not None:
                max_diff = max(max_diff, result['max_abs_diff'])
        
        return {
            'match': all_match,
            'reason': 'Match' if all_match else 'Mismatch in tuple/list',
            'max_abs_diff': max_diff if max_diff > 0 else None,
            'mean_diff': None,
            'shape': None,
            'results': results,
        }
    
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        # If both are dicts, compare them as dicts
        if isinstance(tensor1, dict) and isinstance(tensor2, dict):
            # Compare dict contents
            if set(tensor1.keys()) != set(tensor2.keys()):
                return {
                    'match': False,
                    'reason': f'Dict keys mismatch: {set(tensor1.keys())} vs {set(tensor2.keys())}',
                    'max_abs_diff': None,
                    'mean_diff': None,
                    'shape': None,
                }
            
            all_match = True
            max_diff = 0.0
            for key in tensor1.keys():
                if isinstance(tensor1[key], torch.Tensor) and isinstance(tensor2[key], torch.Tensor):
                    comp = compare_tensors(tensor1[key], tensor2[key], f'{name}.{key}', atol, rtol)
                    if not comp['match']:
                        all_match = False
                    if comp.get('max_abs_diff') is not None:
                        max_diff = max(max_diff, comp['max_abs_diff'])
                elif tensor1[key] != tensor2[key]:
                    all_match = False
            
            return {
                'match': all_match,
                'reason': 'Match' if all_match else f'Max abs diff: {max_diff:.2e}' if max_diff > 0 else 'Dict values mismatch',
                'max_abs_diff': max_diff if max_diff > 0 else None,
                'mean_diff': None,
                'shape': None,
            }
        
        # For other non-tensor types, check equality
        return {
            'match': tensor1 == tensor2,
            'reason': f'Type mismatch: {type(tensor1)} vs {type(tensor2)}' if type(tensor1) != type(tensor2) else f'Value mismatch: {tensor1} vs {tensor2}',
            'max_abs_diff': None,
            'mean_diff': None,
            'shape': None,
        }
    
    if tensor1.shape != tensor2.shape:
        return {
            'match': False,
            'reason': f'Shape mismatch: {tensor1.shape} vs {tensor2.shape}',
            'max_abs_diff': None,
            'mean_diff': None,
            'shape': (list(tensor1.shape), list(tensor2.shape)),
        }
    
    # Handle integer types (Long, Int, etc.) - check equality directly
    if tensor1.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.long):
        # For integer types, check exact equality
        is_equal = torch.equal(tensor1, tensor2)
        max_abs_diff = (tensor1.int() - tensor2.int()).abs().max().item() if not is_equal else 0.0
        mean_diff = None
        std_diff = None
        return {
            'match': is_equal,
            'reason': 'Match' if is_equal else f'Max abs diff: {max_abs_diff}',
            'max_abs_diff': max_abs_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'shape': list(tensor1.shape),
            'dtype': str(tensor1.dtype),
        }
    
    # Convert to same dtype if needed (for floating point types)
    if tensor1.dtype != tensor2.dtype:
        if tensor1.dtype == torch.float64 or tensor2.dtype == torch.float64:
            tensor1 = tensor1.double()
            tensor2 = tensor2.double()
        elif tensor1.dtype == torch.float32 or tensor2.dtype == torch.float32:
            tensor1 = tensor1.float()
            tensor2 = tensor2.float()
    
    diff = tensor1 - tensor2
    max_abs_diff = diff.abs().max().item()
    
    # For floating point types, compute mean and std
    try:
        mean_diff = diff.mean().item()
    except RuntimeError:
        mean_diff = None
    
    try:
        std_diff = diff.std().item()
    except RuntimeError:
        # Handle case where std() fails (e.g., single element tensor)
        std_diff = None
    
    is_close = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
    
    return {
        'match': is_close,
        'reason': 'Match' if is_close else f'Max abs diff: {max_abs_diff:.2e}',
        'max_abs_diff': max_abs_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'shape': list(tensor1.shape),
        'dtype': str(tensor1.dtype),
    }


def normalize_layer_name(name: str, prefix: str = "") -> str:
    """Normalize layer name by removing common prefixes."""
    # Remove common prefixes
    prefixes_to_remove = ['model.', 'module.', 'encoder.', 'decoder.', 'preprocessor.', 'quantizer.']
    for prefix_to_remove in prefixes_to_remove:
        if name.startswith(prefix_to_remove):
            name = name[len(prefix_to_remove):]
    
    # Remove prefix if specified
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    
    return name


def compare_model_structures(
    nemo_structure: Dict[str, Any],
    nest_structure: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare model structures."""
    print("\n" + "="*80)
    print("Model Structure Comparison")
    print("="*80)
    
    nemo_params = nemo_structure.get('parameters', {})
    nest_params = nest_structure.get('parameters', {})
    
    # Normalize parameter names
    nemo_params_normalized = {
        normalize_layer_name(name): (name, info) 
        for name, info in nemo_params.items()
    }
    nest_params_normalized = {
        normalize_layer_name(name): (name, info) 
        for name, info in nest_params.items()
    }
    
    # Find common parameters
    common_params = set(nemo_params_normalized.keys()) & set(nest_params_normalized.keys())
    nemo_only = set(nemo_params_normalized.keys()) - set(nest_params_normalized.keys())
    nest_only = set(nest_params_normalized.keys()) - set(nemo_params_normalized.keys())
    
    print(f"\nTotal parameters:")
    print(f"  NeMo: {len(nemo_params)}")
    print(f"  nest_ssl_project: {len(nest_params)}")
    print(f"  Common: {len(common_params)}")
    print(f"  NeMo only: {len(nemo_only)}")
    print(f"  nest_ssl_project only: {len(nest_only)}")
    
    # Compare common parameters
    shape_matches = 0
    dtype_matches = 0
    
    for param_name in sorted(common_params):
        nemo_name, nemo_info = nemo_params_normalized[param_name]
        nest_name, nest_info = nest_params_normalized[param_name]
        
        nemo_shape = nemo_info.get('shape', [])
        nest_shape = nest_info.get('shape', [])
        
        if nemo_shape == nest_shape:
            shape_matches += 1
        else:
            print(f"\n  Shape mismatch for {param_name}:")
            print(f"    NeMo: {nemo_shape}")
            print(f"    nest: {nest_shape}")
        
        nemo_dtype = nemo_info.get('dtype', '')
        nest_dtype = nest_info.get('dtype', '')
        
        if nemo_dtype == nest_dtype:
            dtype_matches += 1
        else:
            print(f"\n  Dtype mismatch for {param_name}:")
            print(f"    NeMo: {nemo_dtype}")
            print(f"    nest: {nest_dtype}")
    
    print(f"\nParameter comparison:")
    print(f"  Shape matches: {shape_matches}/{len(common_params)}")
    print(f"  Dtype matches: {dtype_matches}/{len(common_params)}")
    
    if nemo_only:
        print(f"\n  NeMo-only parameters (first 10):")
        for param_name in sorted(list(nemo_only))[:10]:
            print(f"    {param_name}")
    
    if nest_only:
        print(f"\n  nest_ssl_project-only parameters (first 10):")
        for param_name in sorted(list(nest_only))[:10]:
            print(f"    {param_name}")
    
    return {
        'total_nemo_params': len(nemo_params),
        'total_nest_params': len(nest_params),
        'common_params': len(common_params),
        'nemo_only': len(nemo_only),
        'nest_only': len(nest_only),
        'shape_matches': shape_matches,
        'dtype_matches': dtype_matches,
    }


def compare_step_outputs(
    nemo_step_dir: Path,
    nest_step_dir: Path,
    step: int,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Dict[str, Any]:
    """Compare outputs for a specific step."""
    print("\n" + "="*80)
    print(f"Step {step} Output Comparison")
    print("="*80)
    
    results = {
        'step': step,
        'forward_output': None,
        'loss': None,
        'parameter_gradients': None,
        'layer_outputs': None,
    }
    
    # Compare batch data first (to check if inputs are the same)
    batch_comparison = None
    if (nemo_step_dir / 'batch.pt').exists() and (nest_step_dir / 'batch.pt').exists():
        try:
            import sys
            import importlib
            import pickle
            
            # Add project root to Python path
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent  # Go up from tools/ to nest_ssl_project/
            parent_root = project_root.parent  # Go up to Nemo_nest/
            
            # Add both paths to sys.path
            for path in [str(project_root), str(parent_root)]:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            # Try to add path mappings for pickle to find modules
            # First try importing as nest_ssl_project.data (when run from parent directory)
            try:
                sys.modules['data'] = importlib.import_module('nest_ssl_project.data')
                sys.modules['data.ssl_dataset'] = importlib.import_module('nest_ssl_project.data.ssl_dataset')
                sys.modules['data.audio_to_text'] = importlib.import_module('nest_ssl_project.data.audio_to_text')
                sys.modules['data.audio_to_text_dataset'] = importlib.import_module('nest_ssl_project.data.audio_to_text_dataset')
            except ModuleNotFoundError:
                # If that fails, try importing as data (when run from nest_ssl_project directory)
                try:
                    sys.modules['data'] = importlib.import_module('data')
                    sys.modules['data.ssl_dataset'] = importlib.import_module('data.ssl_dataset')
                    sys.modules['data.audio_to_text'] = importlib.import_module('data.audio_to_text')
                    sys.modules['data.audio_to_text_dataset'] = importlib.import_module('data.audio_to_text_dataset')
                except ModuleNotFoundError:
                    # If still can't import, we'll try loading anyway and handle errors
                    pass
            
            # Use weights_only=False for PyTorch 2.6+ compatibility with custom classes
            nemo_batch = torch.load(nemo_step_dir / 'batch.pt', map_location='cpu', weights_only=False)
            nest_batch = torch.load(nest_step_dir / 'batch.pt', map_location='cpu', weights_only=False)
            
            # Extract actual tensor data from batch objects
            # Handle dict (saved format), AudioNoiseBatch, or tuple/list
            def extract_tensors(batch_obj):
                """Extract tensor data from various batch formats."""
                if isinstance(batch_obj, dict):
                    # Saved as dict with keys like 'batch_0', 'batch_1', or 'signal', 'length', etc.
                    tensors = {}
                    for k, v in batch_obj.items():
                        if isinstance(v, torch.Tensor):
                            tensors[k] = v
                    return tensors if tensors else None
                elif hasattr(batch_obj, '__dict__') or hasattr(type(batch_obj), '__annotations__'):
                    # Custom class like AudioNoiseBatch (dataclass or NamedTuple)
                    # AudioNoiseBatch has: audio, audio_len, noise, noise_len, noisy_audio, noisy_audio_len
                    tensors = {}
                    # Try common attribute names
                    for attr in ['audio', 'audio_signal', 'signal', 'signal_len', 'audio_len', 
                                 'noise', 'noise_len', 'noisy_audio', 'noisy_audio_len',
                                 'targets', 'target_lengths']:
                        if hasattr(batch_obj, attr):
                            val = getattr(batch_obj, attr)
                            if isinstance(val, torch.Tensor):
                                tensors[attr] = val
                    # Also try __dict__ if available
                    if hasattr(batch_obj, '__dict__'):
                        for k, v in batch_obj.__dict__.items():
                            if isinstance(v, torch.Tensor) and k not in tensors:
                                tensors[k] = v
                    return tensors if tensors else None
                elif isinstance(batch_obj, (list, tuple)):
                    # Tuple/list format: (signal, signal_len, targets, target_lengths)
                    tensors = {}
                    for i, item in enumerate(batch_obj):
                        if isinstance(item, torch.Tensor):
                            tensors[f'item_{i}'] = item
                    return tensors if tensors else None
                elif isinstance(batch_obj, torch.Tensor):
                    return {'tensor': batch_obj}
                return None
            
            nemo_tensors = extract_tensors(nemo_batch)
            nest_tensors = extract_tensors(nest_batch)
            
            if nemo_tensors and nest_tensors:
                # Compare each tensor field
                all_match = True
                max_diff = 0.0
                common_keys = set(nemo_tensors.keys()) & set(nest_tensors.keys())
                nemo_only = set(nemo_tensors.keys()) - set(nest_tensors.keys())
                nest_only = set(nest_tensors.keys()) - set(nemo_tensors.keys())
                
                if nemo_only or nest_only:
                    batch_comparison = {
                        'match': False,
                        'reason': f'Key mismatch: NeMo only {nemo_only}, nest only {nest_only}',
                        'max_abs_diff': None,
                        'mean_diff': None,
                        'std_diff': None,
                    }
                else:
                    for key in common_keys:
                        comp = compare_tensors(nemo_tensors[key], nest_tensors[key], f'batch_{key}', atol, rtol)
                        if not comp['match']:
                            all_match = False
                        if comp.get('max_abs_diff') is not None:
                            max_diff = max(max_diff, comp['max_abs_diff'])
                    
                    batch_comparison = {
                        'match': all_match,
                        'reason': 'Match' if all_match else f'Max abs diff: {max_diff:.2e}',
                        'max_abs_diff': max_diff if max_diff > 0 else None,
                        'mean_diff': None,
                        'std_diff': None,
                    }
            else:
                # Fallback: try direct comparison (handles dict, tuple, etc.)
                batch_comparison = compare_tensors(nemo_batch, nest_batch, 'batch_input', atol, rtol)
            
            print(f"\nBatch Input Data:")
            print(f"  Match: {batch_comparison['match']}")
            print(f"  Reason: {batch_comparison['reason']}")
            
            # Show detailed comparison if available
            if nemo_tensors and nest_tensors:
                print(f"\n  Batch fields comparison:")
                common_keys = set(nemo_tensors.keys()) & set(nest_tensors.keys())
                nemo_only = set(nemo_tensors.keys()) - set(nest_tensors.keys())
                nest_only = set(nest_tensors.keys()) - set(nemo_tensors.keys())
                
                if nemo_only:
                    print(f"    NeMo only keys: {sorted(nemo_only)}")
                if nest_only:
                    print(f"    nest only keys: {sorted(nest_only)}")
                
                print(f"    Common keys: {sorted(common_keys)}")
                print(f"\n  Field-by-field comparison:")
                for key in sorted(common_keys):
                    nemo_val = nemo_tensors[key]
                    nest_val = nest_tensors[key]
                    if isinstance(nemo_val, torch.Tensor) and isinstance(nest_val, torch.Tensor):
                        comp = compare_tensors(nemo_val, nest_val, f'batch.{key}', atol, rtol)
                        match_str = "[OK]" if comp['match'] else "[FAIL]"
                        print(f"    {match_str} {key}:")
                        print(f"      NeMo shape: {list(nemo_val.shape)}, dtype: {nemo_val.dtype}")
                        print(f"      nest shape: {list(nest_val.shape)}, dtype: {nest_val.dtype}")
                        if not comp['match']:
                            if comp.get('max_abs_diff') is not None:
                                print(f"      Max abs diff: {comp['max_abs_diff']:.2e}")
                            if comp.get('mean_diff') is not None:
                                print(f"      Mean diff: {comp['mean_diff']:.2e}")
                            # Show sample values for debugging
                            if nemo_val.numel() > 0 and nest_val.numel() > 0:
                                print(f"      NeMo sample (first 5): {nemo_val.flatten()[:5].tolist()}")
                                print(f"      nest sample (first 5): {nest_val.flatten()[:5].tolist()}")
                    else:
                        match_str = "[OK]" if nemo_val == nest_val else "[FAIL]"
                        print(f"    {match_str} {key}: {type(nemo_val).__name__} vs {type(nest_val).__name__}")
            
            if batch_comparison.get('max_abs_diff') is not None:
                print(f"\n  Overall max abs diff: {batch_comparison['max_abs_diff']:.2e}")
                if batch_comparison.get('mean_diff') is not None:
                    print(f"  Overall mean diff: {batch_comparison['mean_diff']:.2e}")
        except Exception as e:
            print(f"\nBatch Input Data: Could not compare ({e})")
            import traceback
            traceback.print_exc()
    
    # Load forward output
    nemo_forward = torch.load(nemo_step_dir / 'forward_output.pt', map_location='cpu')
    nest_forward = torch.load(nest_step_dir / 'forward_output.pt', map_location='cpu')
    
    forward_comparison = compare_tensors(nemo_forward, nest_forward, 'forward_output', atol, rtol)
    results['forward_output'] = forward_comparison
    results['batch'] = batch_comparison
    
    print(f"\nForward Output:")
    print(f"  Match: {forward_comparison['match']}")
    print(f"  Reason: {forward_comparison['reason']}")
    if forward_comparison.get('max_abs_diff') is not None:
        print(f"  Max abs diff: {forward_comparison['max_abs_diff']:.2e}")
        print(f"  Mean diff: {forward_comparison['mean_diff']:.2e}")
        print(f"  Std diff: {forward_comparison['std_diff']:.2e}")
    
    # Load loss
    nemo_loss = torch.load(nemo_step_dir / 'loss.pt', map_location='cpu')
    nest_loss = torch.load(nest_step_dir / 'loss.pt', map_location='cpu')
    
    loss_comparison = compare_tensors(nemo_loss, nest_loss, 'loss', atol, rtol)
    results['loss'] = loss_comparison
    
    print(f"\nLoss:")
    print(f"  Match: {loss_comparison['match']}")
    print(f"  Reason: {loss_comparison['reason']}")
    if loss_comparison.get('max_abs_diff') is not None:
        print(f"  Max abs diff: {loss_comparison['max_abs_diff']:.2e}")
        print(f"  Mean diff: {loss_comparison['mean_diff']:.2e}")
        print(f"  NeMo loss: {nemo_loss.item() if isinstance(nemo_loss, torch.Tensor) else nemo_loss}")
        print(f"  nest loss: {nest_loss.item() if isinstance(nest_loss, torch.Tensor) else nest_loss}")
    
    # Load parameter gradients
    nemo_grads = torch.load(nemo_step_dir / 'parameter_gradients.pt', map_location='cpu')
    nest_grads = torch.load(nest_step_dir / 'parameter_gradients.pt', map_location='cpu')
    
    # Normalize parameter names
    nemo_grads_normalized = {
        normalize_layer_name(name): (name, grad) 
        for name, grad in nemo_grads.items()
    }
    nest_grads_normalized = {
        normalize_layer_name(name): (name, grad) 
        for name, grad in nest_grads.items()
    }
    
    common_grads = set(nemo_grads_normalized.keys()) & set(nest_grads_normalized.keys())
    
    grad_matches = {}
    grad_match_count = 0
    grad_total_count = 0
    
    print(f"\nParameter Gradients:")
    print(f"  NeMo params: {len(nemo_grads)}")
    print(f"  nest params: {len(nest_grads)}")
    print(f"  Common params: {len(common_grads)}")
    
    for param_name in sorted(common_grads):
        nemo_name, nemo_grad = nemo_grads_normalized[param_name]
        nest_name, nest_grad = nest_grads_normalized[param_name]
        
        grad_total_count += 1
        grad_comparison = compare_tensors(nemo_grad, nest_grad, f'grad_{param_name}', atol, rtol)
        grad_matches[param_name] = grad_comparison
        
        if grad_comparison['match']:
            grad_match_count += 1
        else:
            if grad_total_count <= 10:  # Print first 10 mismatches
                print(f"\n  Mismatch: {param_name}")
                print(f"    {grad_comparison['reason']}")
                if grad_comparison.get('max_abs_diff') is not None:
                    print(f"    Max abs diff: {grad_comparison['max_abs_diff']:.2e}")
    
    print(f"\n  Gradient matches: {grad_match_count}/{grad_total_count} ({grad_match_count/grad_total_count*100:.2f}%)")
    results['parameter_gradients'] = {
        'total': grad_total_count,
        'matches': grad_match_count,
        'match_rate': grad_match_count / grad_total_count if grad_total_count > 0 else 0.0,
        'details': grad_matches,
    }
    
    # Load layer outputs
    with open(nemo_step_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_step_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Normalize layer names
    nemo_layers_normalized = {
        normalize_layer_name(name): (name, data) 
        for name, data in nemo_layers.items()
    }
    nest_layers_normalized = {
        normalize_layer_name(name): (name, data) 
        for name, data in nest_layers.items()
    }
    
    common_layers = set(nemo_layers_normalized.keys()) & set(nest_layers_normalized.keys())
    
    layer_matches = {}
    layer_match_count = 0
    layer_total_count = 0
    
    print(f"\nLayer Outputs:")
    print(f"  NeMo layers: {len(nemo_layers)}")
    print(f"  nest layers: {len(nest_layers)}")
    print(f"  Common layers: {len(common_layers)}")
    
    # Find the first layer with mismatch to identify where the problem starts
    first_mismatch_layer = None
    layer_match_count = 0
    
    # Sort layers to process in order (preprocessor -> encoder -> decoder)
    def layer_sort_key(name):
        """Sort layers by processing order."""
        if 'preprocessor' in name.lower() or 'featurizer' in name.lower():
            return (0, name)
        elif 'encoder' in name.lower():
            return (1, name)
        elif 'decoder' in name.lower():
            return (2, name)
        else:
            return (3, name)
    
    sorted_common_layers = sorted(common_layers, key=layer_sort_key)
    
    print(f"\n  Layer-by-layer comparison (in processing order):")
    for layer_name in sorted_common_layers:
        nemo_name, nemo_data = nemo_layers_normalized[layer_name]
        nest_name, nest_data = nest_layers_normalized[layer_name]
        
        # Compare forward outputs
        nemo_output = nemo_data.get('forward_outputs')
        nest_output = nest_data.get('forward_outputs')
        
        layer_total_count += 1
        output_comparison = compare_tensors(nemo_output, nest_output, f'layer_{layer_name}_output', atol, rtol)
        
        layer_matches[layer_name] = {
            'forward_output': output_comparison,
        }
        
        if output_comparison['match']:
            layer_match_count += 1
            match_str = "[OK]"
        else:
            match_str = "[FAIL]"
            if first_mismatch_layer is None:
                first_mismatch_layer = layer_name
        
        # Show all important layers (preprocessor, encoder layers, decoder) and mismatches
        is_important = ('preprocessor' in layer_name.lower() or 'featurizer' in layer_name.lower() or 
                       'encoder' in layer_name.lower() or 'decoder' in layer_name.lower() or
                       layer_name.startswith('layers.0') or not output_comparison['match'])
        
        if is_important:
            print(f"    {match_str} {layer_name}")
            if output_comparison.get('reason'):
                print(f"      {output_comparison['reason']}")
            if output_comparison.get('max_abs_diff') is not None:
                print(f"      Max abs diff: {output_comparison['max_abs_diff']:.2e}")
            if output_comparison.get('mean_diff') is not None:
                print(f"      Mean diff: {output_comparison['mean_diff']:.2e}")
            if isinstance(nemo_output, torch.Tensor) and isinstance(nest_output, torch.Tensor):
                print(f"      NeMo shape: {list(nemo_output.shape)}, dtype: {nemo_output.dtype}")
                print(f"      nest shape: {list(nest_output.shape)}, dtype: {nest_output.dtype}")
                if not output_comparison['match'] and nemo_output.numel() > 0 and nest_output.numel() > 0:
                    print(f"      NeMo sample (first 5): {nemo_output.flatten()[:5].tolist()}")
                    print(f"      nest sample (first 5): {nest_output.flatten()[:5].tolist()}")
    
    if first_mismatch_layer:
        print(f"\n  ⚠️  FIRST MISMATCH at layer: {first_mismatch_layer}")
        print(f"     This indicates the problem starts at this layer or earlier.")
    
    print(f"\n  Layer output matches: {layer_match_count}/{layer_total_count} ({layer_match_count/layer_total_count*100:.2f}%)")
    results['layer_outputs'] = {
        'total': layer_total_count,
        'matches': layer_match_count,
        'match_rate': layer_match_count / layer_total_count if layer_total_count > 0 else 0.0,
        'details': layer_matches,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare NeMo and nest_ssl_project outputs')
    parser.add_argument('--nemo_output_dir', type=str, required=True,
                        help='Directory containing saved NeMo outputs')
    parser.add_argument('--nest_output_dir', type=str, required=True,
                        help='Directory containing saved nest_ssl_project outputs')
    parser.add_argument('--step', type=int, default=None,
                        help='Step number to compare (default: None, auto-detect from available steps)')
    parser.add_argument('--atol', type=float, default=1e-5,
                        help='Absolute tolerance for comparison (default: 1e-5)')
    parser.add_argument('--rtol', type=float, default=1e-5,
                        help='Relative tolerance for comparison (default: 1e-5)')
    parser.add_argument('--save_comparison', type=str, default=None,
                        help='Path to save comparison results (optional)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save output to txt file (default: comparison_results_step_{step}.txt)')
    
    args = parser.parse_args()
    
    # Resolve paths (handle both relative and absolute paths)
    nemo_output_dir = Path(args.nemo_output_dir).resolve()
    nest_output_dir = Path(args.nest_output_dir).resolve()
    
    if not nemo_output_dir.exists():
        print(f"Error: NeMo output directory does not exist: {nemo_output_dir}")
        print(f"  Resolved path: {nemo_output_dir.absolute()}")
        sys.exit(1)
    
    if not nest_output_dir.exists():
        print(f"Error: nest_ssl_project output directory does not exist: {nest_output_dir}")
        print(f"  Resolved path: {nest_output_dir.absolute()}")
        sys.exit(1)
    
    # Setup output file - will be set after step is determined
    output_file_path = None
    output_file_handle = None
    original_stdout = sys.stdout
    
    if args.output_file:
        output_file_path = Path(args.output_file)
    
    # Create a Tee class to write to both stdout and file
    class Tee:
        def __init__(self, *files):
            self.files = files
        
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Load model structures
    print("Loading model structures...")
    with open(nemo_output_dir / 'model_structure.pkl', 'rb') as f:
        nemo_structure = pickle.load(f)
    with open(nest_output_dir / 'model_structure.pkl', 'rb') as f:
        nest_structure = pickle.load(f)
    
    # Compare model structures
    structure_comparison = compare_model_structures(nemo_structure, nest_structure)
    
    # Auto-detect step if not specified
    if args.step is None:
        # Find available step directories
        nemo_steps = []
        nest_steps = []
        
        if nemo_output_dir.exists():
            for item in nemo_output_dir.iterdir():
                if item.is_dir() and item.name.startswith('step_'):
                    try:
                        step_num = int(item.name.split('_')[1])
                        nemo_steps.append(step_num)
                    except (ValueError, IndexError):
                        pass
        
        if nest_output_dir.exists():
            for item in nest_output_dir.iterdir():
                if item.is_dir() and item.name.startswith('step_'):
                    try:
                        step_num = int(item.name.split('_')[1])
                        nest_steps.append(step_num)
                    except (ValueError, IndexError):
                        pass
        
        # Find common steps
        common_steps = sorted(set(nemo_steps) & set(nest_steps))
        
        if not common_steps:
            print(f"Error: No common step directories found!")
            print(f"  NeMo steps: {sorted(nemo_steps)}")
            print(f"  nest_ssl_project steps: {sorted(nest_steps)}")
            sys.exit(1)
        
        # Use the first common step (lowest step number)
        args.step = common_steps[0]
        print(f"Auto-detected step: {args.step} (available steps: {common_steps})")
    
    # Set output file path if not specified but step is known
    if not output_file_path and args.step is not None:
        output_file_path = nest_output_dir.parent / f'comparison_results_step_{args.step}.txt'
    
    # Redirect stdout to both console and file if output file is specified
    if output_file_path:
        print(f"Saving comparison results to: {output_file_path}")
        output_file_handle = open(output_file_path, 'w', encoding='utf-8')
        sys.stdout = Tee(original_stdout, output_file_handle)
    
    # Compare step outputs
    nemo_step_dir = nemo_output_dir / f"step_{args.step}"
    nest_step_dir = nest_output_dir / f"step_{args.step}"
    
    if not nemo_step_dir.exists():
        print(f"Error: NeMo step directory does not exist: {nemo_step_dir}")
        print(f"  Resolved path: {nemo_step_dir.absolute()}")
        print(f"  Available directories in {nemo_output_dir}:")
        if nemo_output_dir.exists():
            for item in nemo_output_dir.iterdir():
                print(f"    - {item.name} ({'dir' if item.is_dir() else 'file'})")
        else:
            print(f"    (directory does not exist)")
        sys.exit(1)
    
    if not nest_step_dir.exists():
        print(f"Error: nest_ssl_project step directory does not exist: {nest_step_dir}")
        print(f"  Resolved path: {nest_step_dir.absolute()}")
        print(f"  Available directories in {nest_output_dir}:")
        if nest_output_dir.exists():
            for item in nest_output_dir.iterdir():
                print(f"    - {item.name} ({'dir' if item.is_dir() else 'file'})")
        else:
            print(f"    (directory does not exist)")
        sys.exit(1)
    
    step_comparison = compare_step_outputs(
        nemo_step_dir, nest_step_dir, args.step, args.atol, args.rtol
    )
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    print(f"\nModel Structure:")
    print(f"  Common parameters: {structure_comparison['common_params']}")
    print(f"  Shape matches: {structure_comparison['shape_matches']}/{structure_comparison['common_params']}")
    print(f"  Dtype matches: {structure_comparison['dtype_matches']}/{structure_comparison['common_params']}")
    
    print(f"\nStep {args.step} Outputs:")
    print(f"  Forward output match: {step_comparison['forward_output']['match']}")
    print(f"  Loss match: {step_comparison['loss']['match']}")
    
    if step_comparison['parameter_gradients']:
        grad_match_rate = step_comparison['parameter_gradients']['match_rate']
        print(f"  Parameter gradients match rate: {grad_match_rate*100:.2f}%")
    
    if step_comparison['layer_outputs']:
        layer_match_rate = step_comparison['layer_outputs']['match_rate']
        print(f"  Layer outputs match rate: {layer_match_rate*100:.2f}%")
    
    # Close file if opened
    if output_file_handle:
        sys.stdout.flush()
        sys.stdout = original_stdout
        output_file_handle.close()
        print(f"\n[OK] Comparison results saved to: {output_file_path}")
    
    # Save comparison results if requested
    if args.save_comparison:
        comparison_results = {
            'structure_comparison': structure_comparison,
            'step_comparison': step_comparison,
            'atol': args.atol,
            'rtol': args.rtol,
        }
        
        save_path = Path(args.save_comparison)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(comparison_results, f)
        
        print(f"\nComparison results saved to: {save_path}")


if __name__ == "__main__":
    main()

