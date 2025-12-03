#!/usr/bin/env python3
"""
Extract actual PyTorch operators (like conv2d, matmul, add, etc.) from model execution.
Uses PyTorch profiler to capture all operators.
"""

import torch
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Set
import sys


def extract_operators_with_profiler(model: torch.nn.Module, dummy_input: torch.Tensor) -> Dict[str, Any]:
    """Extract operators using PyTorch profiler."""
    operators = defaultdict(lambda: {
        "count": 0,
        "input_shapes": [],
        "output_shapes": [],
        "call_sites": [],
    })
    
    # Use PyTorch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Extract operator information from profiler
    for event in prof.events():
        if event.key in ["aten::conv2d", "aten::conv1d", "aten::matmul", "aten::add", 
                         "aten::mul", "aten::relu", "aten::gelu", "aten::silu",
                         "aten::batch_norm", "aten::layer_norm", "aten::dropout",
                         "aten::embedding", "aten::transpose", "aten::view", "aten::reshape",
                         "aten::cat", "aten::stack", "aten::split", "aten::chunk",
                         "aten::unsqueeze", "aten::squeeze", "aten::expand", "aten::repeat",
                         "aten::pad", "aten::slice", "aten::select", "aten::index",
                         "aten::gather", "aten::scatter", "aten::masked_fill", "aten::where",
                         "aten::softmax", "aten::log_softmax", "aten::sigmoid", "aten::tanh",
                         "aten::max_pool2d", "aten::avg_pool2d", "aten::adaptive_avg_pool2d",
                         "aten::stft", "aten::fft", "aten::rfft", "aten::irfft",
                         "aten::abs", "aten::sqrt", "aten::pow", "aten::exp", "aten::log",
                         "aten::sum", "aten::mean", "aten::std", "aten::var",
                         "aten::max", "aten::min", "aten::argmax", "aten::argmin",
                         "aten::clamp", "aten::clamp_min", "aten::clamp_max",
                         "aten::floor", "aten::ceil", "aten::round", "aten::trunc",
                         "aten::eq", "aten::ne", "aten::lt", "aten::le", "aten::gt", "aten::ge",
                         "aten::and", "aten::or", "aten::not", "aten::xor",
                         "aten::ones", "aten::zeros", "aten::ones_like", "aten::zeros_like",
                         "aten::rand", "aten::randn", "aten::uniform", "aten::normal",
                         "aten::arange", "aten::linspace", "aten::meshgrid",
                         "aten::to", "aten::cuda", "aten::cpu", "aten::type_as",
                         "aten::detach", "aten::clone", "aten::contiguous",
                         "aten::backward", "aten::grad", "aten::requires_grad_",
                         ]:
            op_name = event.key.replace("aten::", "")
            operators[op_name]["count"] += 1
            
            # Extract shapes if available
            if hasattr(event, 'input_shapes') and event.input_shapes:
                operators[op_name]["input_shapes"].extend(event.input_shapes)
            if hasattr(event, 'output_shapes') and event.output_shapes:
                operators[op_name]["output_shapes"].extend(event.output_shapes)
            
            # Extract call site
            if hasattr(event, 'stack') and event.stack:
                operators[op_name]["call_sites"].append(event.stack[0] if event.stack else "")
    
    return dict(operators)


def extract_operators_from_torchscript(model: torch.nn.Module, dummy_input: torch.Tensor) -> Dict[str, Any]:
    """Extract operators from TorchScript graph."""
    operators = defaultdict(lambda: {
        "count": 0,
        "nodes": [],
    })
    
    try:
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Get the graph
        graph = traced_model.graph
        
        # Extract all nodes (operators)
        for node in graph.nodes():
            op_kind = node.kind()
            
            # Extract operator name (remove namespace if present)
            if "::" in op_kind:
                op_name = op_kind.split("::")[-1]
            else:
                op_name = op_kind
            
            operators[op_name]["count"] += 1
            
            # Extract node information
            node_info = {
                "kind": op_kind,
                "inputs": [],
                "outputs": [],
            }
            
            # Extract input types/shapes
            for inp in node.inputs():
                if hasattr(inp, 'type'):
                    node_info["inputs"].append(str(inp.type()))
            
            # Extract output types/shapes
            for out in node.outputs():
                if hasattr(out, 'type'):
                    node_info["outputs"].append(str(out.type()))
            
            operators[op_name]["nodes"].append(node_info)
    
    except Exception as e:
        print(f"Warning: TorchScript tracing failed: {e}")
        return {}
    
    return dict(operators)


def extract_operators_from_saved_outputs(output_dir: Path) -> Dict[str, Any]:
    """Extract operators from saved layer outputs by analyzing tensor operations."""
    import pickle
    
    layer_outputs_path = output_dir / 'layer_outputs.pkl'
    if not layer_outputs_path.exists():
        return {}
    
    with open(layer_outputs_path, 'rb') as f:
        layer_outputs = pickle.load(f)
    
    operators = defaultdict(lambda: {
        "count": 0,
        "shapes": [],
    })
    
    # Analyze tensor operations from shapes
    for layer_name, layer_data in layer_outputs.items():
        if not isinstance(layer_data, dict):
            continue
        
        # Get input/output shapes
        forward_inputs = layer_data.get('all_forward_inputs', [])
        forward_outputs = layer_data.get('all_forward_outputs', [])
        
        # Infer operators from shape transformations
        input_shapes = []
        output_shapes = []
        
        for inp in forward_inputs:
            if isinstance(inp, (list, tuple)):
                for item in inp:
                    if isinstance(item, torch.Tensor):
                        input_shapes.append(list(item.shape))
            elif isinstance(inp, torch.Tensor):
                input_shapes.append(list(inp.shape))
        
        for out in forward_outputs:
            if isinstance(out, (list, tuple)):
                for item in out:
                    if isinstance(item, torch.Tensor):
                        output_shapes.append(list(item.shape))
            elif isinstance(out, torch.Tensor):
                output_shapes.append(list(out.shape))
        
        # Infer operators from shape changes
        if input_shapes and output_shapes:
            for inp_shape, out_shape in zip(input_shapes[:1], output_shapes[:1]):  # Just first pair
                if len(inp_shape) == 4 and len(out_shape) == 4:
                    # Conv2d-like operation
                    operators["conv2d"]["count"] += 1
                elif len(inp_shape) == 2 and len(out_shape) == 2:
                    # Linear/MatMul-like operation
                    operators["matmul"]["count"] += 1
                elif inp_shape == out_shape:
                    # Element-wise operation (add, mul, relu, etc.)
                    operators["elementwise"]["count"] += 1
    
    return dict(operators)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract actual PyTorch operators from model")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, help="Path to saved outputs directory (alternative to model_path)")
    parser.add_argument("--output", type=str, default="pytorch_operators.json", help="Output JSON file")
    parser.add_argument("--method", type=str, choices=["profiler", "torchscript", "saved_outputs"], 
                       default="profiler", help="Extraction method")
    parser.add_argument("--dummy_input_shape", type=str, default="2,16000", 
                       help="Dummy input shape (comma-separated, e.g., '2,16000' for audio)")
    
    args = parser.parse_args()
    
    result = {
        "extraction_method": args.method,
        "operators": {},
    }
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        result["source"] = str(output_dir)
        
        if args.method == "saved_outputs":
            print(f"Extracting operators from saved outputs in {output_dir}...")
            operators = extract_operators_from_saved_outputs(output_dir)
            result["operators"] = operators
        else:
            print("Error: --output_dir only works with --method saved_outputs")
            return
    
    elif args.model_path:
        model_path = Path(args.model_path)
        result["source"] = str(model_path)
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Try to load model (this is a simplified version - you may need to adapt)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("Warning: Model instantiation not fully implemented. Please provide a model instance.")
            return
        
        # For now, we'll need the user to provide a model instance
        # This is a placeholder - you would need to instantiate the actual model
        print("Error: Model loading not fully implemented. Please use --output_dir with saved outputs.")
        return
    
    else:
        print("Error: Either --model_path or --output_dir must be provided")
        return
    
    # Save results
    output_path = Path(args.output)
    print(f"\nSaving results to {output_path}...")
    
    # Convert to JSON-serializable format
    json_result = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_result)
    
    print(f"✓ Saved operator information to {output_path}")
    print(f"  Total unique operators: {len(result['operators'])}")
    
    if result['operators']:
        print(f"  Most common operators:")
        sorted_ops = sorted(result['operators'].items(), 
                          key=lambda x: x[1].get('count', 0), reverse=True)[:20]
        for op_name, op_info in sorted_ops:
            count = op_info.get('count', 0)
            print(f"    {op_name}: {count} occurrences")


if __name__ == '__main__':
    main()

