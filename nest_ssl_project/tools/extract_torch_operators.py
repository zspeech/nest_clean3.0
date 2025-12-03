#!/usr/bin/env python3
"""
Extract all PyTorch operators used in a model and save to JSON.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Set
import sys


def get_torch_operators() -> Set[str]:
    """Get all available PyTorch operators."""
    operators = set()
    
    # Get all torch functions
    for name in dir(torch):
        obj = getattr(torch, name)
        if callable(obj) and not name.startswith('_'):
            operators.add(f"torch.{name}")
    
    # Get all torch.nn modules
    for name in dir(nn):
        obj = getattr(nn, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module) and not name.startswith('_'):
            operators.add(f"nn.{name}")
    
    # Get all torch.nn.functional functions
    import torch.nn.functional as F
    for name in dir(F):
        obj = getattr(F, name)
        if callable(obj) and not name.startswith('_'):
            operators.add(f"F.{name}")
    
    return operators


def extract_model_operators(model: nn.Module, model_name: str = "model") -> Dict[str, Any]:
    """Extract all operators used in a model."""
    operators_used = defaultdict(int)
    module_types = defaultdict(int)
    operator_details = []
    
    def register_hook(name, module):
        """Register forward hook to capture operator usage."""
        def hook(module, input, output):
            module_type = type(module).__name__
            module_path = name
            
            # Count module type
            module_types[module_type] += 1
            
            # Extract operator info
            op_info = {
                "module_type": module_type,
                "module_path": module_path,
                "input_shapes": [],
                "output_shapes": [],
            }
            
            # Extract input shapes
            if isinstance(input, (tuple, list)):
                for inp in input:
                    if isinstance(inp, torch.Tensor):
                        op_info["input_shapes"].append(list(inp.shape))
            elif isinstance(input, torch.Tensor):
                op_info["input_shapes"].append(list(input.shape))
            
            # Extract output shapes
            if isinstance(output, (tuple, list)):
                for out in output:
                    if isinstance(out, torch.Tensor):
                        op_info["output_shapes"].append(list(out.shape))
            elif isinstance(output, torch.Tensor):
                op_info["output_shapes"].append(list(output.shape))
            
            operator_details.append(op_info)
            
            # Count operator
            operators_used[module_type] += 1
        
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if name:  # Skip root module
            hook = module.register_forward_hook(register_hook(name, module))
            hooks.append(hook)
    
    return {
        "operators_used": dict(operators_used),
        "module_types": dict(module_types),
        "operator_details": operator_details,
    }


def extract_static_operators(model: nn.Module) -> Dict[str, Any]:
    """Extract operators from model structure (without forward pass)."""
    operators_used = defaultdict(int)
    module_info = []
    
    for name, module in model.named_modules():
        if name:  # Skip root module
            module_type = type(module).__name__
            operators_used[module_type] += 1
            
            info = {
                "module_type": module_type,
                "module_path": name,
                "parameters": {},
            }
            
            # Extract parameter shapes
            for param_name, param in module.named_parameters(recurse=False):
                info["parameters"][param_name] = {
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "requires_grad": param.requires_grad,
                }
            
            # Extract buffer shapes
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer_name not in info:
                    info["buffers"] = {}
                if "buffers" not in info:
                    info["buffers"] = {}
                info["buffers"][buffer_name] = {
                    "shape": list(buffer.shape),
                    "dtype": str(buffer.dtype),
                }
            
            module_info.append(info)
    
    return {
        "operators_used": dict(operators_used),
        "module_info": module_info,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract PyTorch operators from model")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint (.pt or .pth file)")
    parser.add_argument("--model_class", type=str, help="Model class name (e.g., EncDecDenoiseMaskedTokenPredModel)")
    parser.add_argument("--config_path", type=str, help="Path to config file (if needed to instantiate model)")
    parser.add_argument("--output", type=str, default="torch_operators.json", help="Output JSON file path")
    parser.add_argument("--mode", type=str, choices=["static", "dynamic", "both"], default="static",
                       help="Extraction mode: static (structure only), dynamic (with forward pass), or both")
    parser.add_argument("--dummy_input", action="store_true", help="Use dummy input for dynamic extraction")
    
    args = parser.parse_args()
    
    result = {
        "extraction_mode": args.mode,
        "model_path": args.model_path,
    }
    
    # Get all available PyTorch operators
    all_operators = sorted(get_torch_operators())
    result["all_torch_operators"] = all_operators
    result["total_operators"] = len(all_operators)
    
    # Load model if path provided
    model = None
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # PyTorch Lightning checkpoint
            state_dict = checkpoint['state_dict']
            # Try to instantiate model
            if args.model_class:
                # This would require importing the model class
                print(f"Warning: Model instantiation not implemented. Using state_dict only.")
                result["model_type"] = "checkpoint_with_state_dict"
            else:
                result["model_type"] = "checkpoint_with_state_dict"
        elif isinstance(checkpoint, dict):
            # Assume it's a state_dict
            result["model_type"] = "state_dict"
        else:
            result["model_type"] = "unknown"
            print(f"Warning: Unknown checkpoint format")
    else:
        print("No model path provided. Only listing available operators.")
    
    # Extract operators from model structure
    if model is not None and args.mode in ["static", "both"]:
        print("Extracting static operators from model structure...")
        static_result = extract_static_operators(model)
        result["static_extraction"] = static_result
    
    # Extract operators from forward pass (if model available and mode is dynamic)
    if model is not None and args.mode in ["dynamic", "both"]:
        print("Extracting dynamic operators from forward pass...")
        if args.dummy_input:
            # Create dummy input
            dummy_input = torch.randn(2, 16000)  # Example: batch_size=2, audio_length=16000
            with torch.no_grad():
                try:
                    _ = model(dummy_input)
                    dynamic_result = extract_model_operators(model)
                    result["dynamic_extraction"] = dynamic_result
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    result["dynamic_extraction_error"] = str(e)
        else:
            print("Warning: Dynamic extraction requires --dummy_input flag")
    
    # Save to JSON
    output_path = Path(args.output)
    print(f"\nSaving results to {output_path}...")
    
    # Convert to JSON-serializable format
    def convert_to_json(obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return {
                "type": "tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        elif isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    json_result = convert_to_json(result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(all_operators)} operators to {output_path}")
    
    if model is not None and args.mode in ["static", "both"]:
        if "static_extraction" in result:
            print(f"✓ Found {len(result['static_extraction']['operators_used'])} unique operator types in model")
            print(f"  Most common operators:")
            sorted_ops = sorted(result['static_extraction']['operators_used'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]
            for op, count in sorted_ops:
                print(f"    {op}: {count}")


if __name__ == '__main__':
    main()

