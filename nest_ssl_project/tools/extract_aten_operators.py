#!/usr/bin/env python3
"""
Extract actual ATen operators (PyTorch backend operators) from model.
Uses TorchScript to trace the model and extract all operator calls.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Set
import sys


def extract_aten_operators_from_graph(graph) -> Dict[str, Any]:
    """Extract ATen operators from TorchScript graph."""
    operators = defaultdict(lambda: {
        "count": 0,
        "nodes": [],
    })
    
    def process_node(node):
        """Recursively process a node and its subgraphs."""
        op_kind = node.kind()
        
        # Extract operator name
        if "::" in op_kind:
            namespace, op_name = op_kind.split("::", 1)
        else:
            namespace = "prim"
            op_name = op_kind
        
        # Only track ATen operators (actual PyTorch backend ops)
        if namespace == "aten":
            operators[op_name]["count"] += 1
            
            node_info = {
                "kind": op_kind,
                "inputs": [],
                "outputs": [],
            }
            
            # Extract input information
            for inp in node.inputs():
                inp_info = {
                    "type": str(inp.type()) if hasattr(inp, 'type') else "unknown",
                }
                if hasattr(inp, 'debugName'):
                    inp_info["name"] = inp.debugName()
                node_info["inputs"].append(inp_info)
            
            # Extract output information
            for out in node.outputs():
                out_info = {
                    "type": str(out.type()) if hasattr(out, 'type') else "unknown",
                }
                if hasattr(out, 'debugName'):
                    out_info["name"] = out.debugName()
                node_info["outputs"].append(out_info)
            
            operators[op_name]["nodes"].append(node_info)
        
        # Process subgraphs (for control flow)
        for block in node.blocks():
            for sub_node in block.nodes():
                process_node(sub_node)
    
    # Process all nodes in the graph
    for node in graph.nodes():
        process_node(node)
    
    return dict(operators)


def extract_operators_from_model(model: nn.Module, dummy_input: torch.Tensor) -> Dict[str, Any]:
    """Extract operators by tracing the model with TorchScript."""
    print("Tracing model with TorchScript...")
    
    try:
        model.eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        
        graph = traced_model.graph
        
        operators = extract_aten_operators_from_graph(graph)
        
        return operators
    
    except Exception as e:
        print(f"Error during tracing: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_all_aten_operators() -> List[str]:
    """Get list of all available ATen operators."""
    # This is a comprehensive list of common ATen operators
    aten_operators = [
        # Tensor creation
        "zeros", "ones", "zeros_like", "ones_like", "randn", "rand", "arange", "linspace",
        "empty", "full", "eye", "diag",
        
        # Unary ops
        "abs", "neg", "sqrt", "rsqrt", "exp", "log", "log10", "log2", "sin", "cos", "tan",
        "asin", "acos", "atan", "sinh", "cosh", "tanh", "sigmoid", "relu", "gelu", "silu",
        "floor", "ceil", "round", "trunc", "frac", "reciprocal",
        
        # Binary ops
        "add", "sub", "mul", "div", "pow", "fmod", "remainder", "atan2",
        "max", "min", "maximum", "minimum",
        
        # Comparison ops
        "eq", "ne", "lt", "le", "gt", "ge", "isnan", "isinf", "isfinite",
        
        # Logical ops
        "logical_and", "logical_or", "logical_not", "logical_xor",
        
        # Reduction ops
        "sum", "mean", "std", "var", "prod", "max", "min", "argmax", "argmin",
        "all", "any", "norm", "dist",
        
        # Linear algebra
        "matmul", "mm", "bmm", "addmm", "addbmm", "mv", "dot",
        
        # Convolution
        "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
        "conv1d_backward", "conv2d_backward",
        
        # Pooling
        "max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d",
        "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_avg_pool1d", "adaptive_avg_pool2d",
        
        # Normalization
        "batch_norm", "layer_norm", "group_norm", "instance_norm",
        "local_response_norm",
        
        # Activation
        "relu", "relu6", "leaky_relu", "prelu", "rrelu", "elu", "selu", "gelu", "silu",
        "hardtanh", "hardswish", "mish", "swish",
        
        # Dropout
        "dropout", "feature_dropout", "alpha_dropout",
        
        # Embedding
        "embedding", "embedding_bag",
        
        # Attention
        "scaled_dot_product_attention",
        
        # Shape ops
        "view", "reshape", "flatten", "squeeze", "unsqueeze", "expand", "expand_as",
        "repeat", "tile", "transpose", "permute", "swapaxes", "swapdims",
        
        # Slicing and indexing
        "slice", "select", "index_select", "masked_select", "nonzero", "gather", "scatter",
        "index", "index_add", "index_copy", "index_fill",
        
        # Concatenation and splitting
        "cat", "stack", "hstack", "vstack", "split", "chunk", "unbind",
        
        # Padding
        "pad", "constant_pad_nd", "reflection_pad1d", "reflection_pad2d",
        "replication_pad1d", "replication_pad2d",
        
        # FFT
        "fft_fft", "fft_ifft", "fft_rfft", "fft_irfft", "fft_fft2", "fft_ifft2",
        "stft", "istft",
        
        # Type conversion
        "to", "type_as", "cast", "int", "long", "float", "double", "half", "bool",
        
        # Device ops
        "cpu", "cuda", "to_mkldnn", "contiguous",
        
        # Memory ops
        "clone", "detach", "copy_", "copy", "pin_memory", "unpin_memory",
        
        # Gradient ops
        "requires_grad_", "retain_grad", "set_grad_enabled",
    ]
    
    return sorted(aten_operators)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ATen operators from model")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint (.pt or .pth)")
    parser.add_argument("--model_class", type=str, help="Model class name (for instantiation)")
    parser.add_argument("--dummy_input_shape", type=str, default="2,80,1584",
                       help="Dummy input shape (comma-separated, e.g., '2,80,1584')")
    parser.add_argument("--output", type=str, default="aten_operators.json",
                       help="Output JSON file")
    parser.add_argument("--list_all", action="store_true",
                       help="List all available ATen operators without model")
    
    args = parser.parse_args()
    
    result = {
        "extraction_method": "torchscript",
    }
    
    # List all operators if requested
    if args.list_all:
        all_ops = get_all_aten_operators()
        result["all_aten_operators"] = all_ops
        result["total_operators"] = len(all_ops)
        print(f"Found {len(all_ops)} ATen operators")
    
    # Extract from model if provided
    if args.model_path:
        model_path = Path(args.model_path)
        result["model_path"] = str(model_path)
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Parse dummy input shape
        shape_parts = [int(x) for x in args.dummy_input_shape.split(',')]
        dummy_input = torch.randn(*shape_parts)
        print(f"Using dummy input shape: {dummy_input.shape}")
        
        # Try to instantiate model (simplified - you may need to adapt)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("Warning: Model instantiation required. Please provide model instance.")
            print("For now, listing available operators only.")
            result["note"] = "Model loading not implemented. Use --list_all to see available operators."
        else:
            print("Error: Unsupported checkpoint format")
            return
    
    # Save results
    output_path = Path(args.output)
    print(f"\nSaving results to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ Saved to {output_path}")


if __name__ == '__main__':
    main()

