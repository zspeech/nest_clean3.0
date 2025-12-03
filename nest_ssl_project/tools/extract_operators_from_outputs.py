#!/usr/bin/env python3
"""
Extract all PyTorch operators from saved training outputs (layer_outputs.pkl).
"""

import torch
import pickle
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List


def extract_operators_from_layer_outputs(layer_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract operator information from layer_outputs dictionary."""
    operators = defaultdict(lambda: {
        "count": 0,
        "input_shapes": [],
        "output_shapes": [],
        "module_paths": [],
    })
    
    for layer_name, layer_data in layer_outputs.items():
        if not isinstance(layer_data, dict):
            continue
        
        # Extract module type from layer name (e.g., "encoder.layers.0.conv" -> "Conv2d")
        module_type = layer_name.split('.')[-1]  # Last part is usually the module type
        
        # Get forward inputs/outputs
        forward_inputs = layer_data.get('all_forward_inputs', [])
        forward_outputs = layer_data.get('all_forward_outputs', [])
        
        # Process inputs
        input_shapes = []
        if forward_inputs:
            for inp in forward_inputs:
                if isinstance(inp, (list, tuple)):
                    for item in inp:
                        if isinstance(item, torch.Tensor):
                            input_shapes.append(list(item.shape))
                elif isinstance(inp, torch.Tensor):
                    input_shapes.append(list(inp.shape))
        
        # Process outputs
        output_shapes = []
        if forward_outputs:
            for out in forward_outputs:
                if isinstance(out, (list, tuple)):
                    for item in out:
                        if isinstance(item, torch.Tensor):
                            output_shapes.append(list(item.shape))
                elif isinstance(out, torch.Tensor):
                    output_shapes.append(list(out.shape))
        
        # Update operator info
        if input_shapes or output_shapes:
            operators[module_type]["count"] += 1
            operators[module_type]["module_paths"].append(layer_name)
            if input_shapes:
                operators[module_type]["input_shapes"].extend(input_shapes)
            if output_shapes:
                operators[module_type]["output_shapes"].extend(output_shapes)
    
    return dict(operators)


def get_operator_mapping() -> Dict[str, str]:
    """Map common module names to PyTorch operator names."""
    return {
        "conv": "Conv2d",
        "conv1d": "Conv1d",
        "conv2d": "Conv2d",
        "linear": "Linear",
        "batch_norm": "BatchNorm2d",
        "batch_norm1d": "BatchNorm1d",
        "dropout": "Dropout",
        "relu": "ReLU",
        "gelu": "GELU",
        "silu": "SiLU",
        "swish": "SiLU",
        "activation": "Activation",
        "layer_norm": "LayerNorm",
        "embedding": "Embedding",
        "attention": "MultiHeadAttention",
        "self_attn": "MultiHeadAttention",
        "pos_enc": "PositionalEncoding",
        "feed_forward": "FeedForward",
        "encoder": "Encoder",
        "decoder": "Decoder",
        "preprocessor": "Preprocessor",
        "featurizer": "FilterbankFeatures",
        "quantizer": "VectorQuantizer",
        "mask_processor": "RandomBlockMasking",
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract PyTorch operators from saved training outputs")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Path to saved outputs directory (e.g., saved_nemo_outputs/step_0)")
    parser.add_argument("--output", type=str, default="operators.json", 
                       help="Output JSON file path")
    parser.add_argument("--include_shapes", action="store_true",
                       help="Include input/output shapes in output")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    layer_outputs_path = output_dir / 'layer_outputs.pkl'
    
    if not layer_outputs_path.exists():
        print(f"Error: {layer_outputs_path} does not exist!")
        return
    
    print(f"Loading layer outputs from {layer_outputs_path}...")
    with open(layer_outputs_path, 'rb') as f:
        layer_outputs = pickle.load(f)
    
    print(f"Found {len(layer_outputs)} layers")
    
    # Extract operators
    operators = extract_operators_from_layer_outputs(layer_outputs)
    
    # Get operator mapping
    operator_mapping = get_operator_mapping()
    
    # Build result
    result = {
        "source": str(output_dir),
        "total_layers": len(layer_outputs),
        "unique_operators": len(operators),
        "operators": {},
    }
    
    # Process operators
    for op_name, op_info in operators.items():
        # Try to map to standard PyTorch operator name
        mapped_name = operator_mapping.get(op_name.lower(), op_name)
        
        operator_data = {
            "count": op_info["count"],
            "module_paths": op_info["module_paths"][:10],  # Limit to first 10 paths
            "total_paths": len(op_info["module_paths"]),
        }
        
        if args.include_shapes:
            # Get unique shapes
            unique_input_shapes = []
            seen_input = set()
            for shape in op_info["input_shapes"]:
                shape_tuple = tuple(shape)
                if shape_tuple not in seen_input:
                    seen_input.add(shape_tuple)
                    unique_input_shapes.append(shape)
            
            unique_output_shapes = []
            seen_output = set()
            for shape in op_info["output_shapes"]:
                shape_tuple = tuple(shape)
                if shape_tuple not in seen_output:
                    seen_output.add(shape_tuple)
                    unique_output_shapes.append(shape)
            
            operator_data["input_shapes"] = unique_input_shapes[:10]  # Limit to first 10
            operator_data["output_shapes"] = unique_output_shapes[:10]
            operator_data["total_input_shapes"] = len(unique_input_shapes)
            operator_data["total_output_shapes"] = len(unique_output_shapes)
        
        result["operators"][mapped_name] = operator_data
    
    # Save to JSON
    output_path = Path(args.output)
    print(f"\nSaving results to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved operator information to {output_path}")
    print(f"  Total operators: {len(operators)}")
    print(f"  Most common operators:")
    sorted_ops = sorted(operators.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
    for op_name, op_info in sorted_ops:
        mapped_name = operator_mapping.get(op_name.lower(), op_name)
        print(f"    {mapped_name}: {op_info['count']} occurrences")


if __name__ == '__main__':
    main()

