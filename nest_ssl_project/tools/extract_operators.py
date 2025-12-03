#!/usr/bin/env python3
"""
Extract actual PyTorch ATen operators from a model.
Usage:
    python extract_operators.py --model <model_instance> --dummy_input <tensor> --output operators.json
    Or import this module and call extract_operators_from_model(model, dummy_input)
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple


def extract_operators_from_model(model: nn.Module, dummy_input: torch.Tensor, 
                                verbose: bool = False) -> Dict[str, Any]:
    """
    Extract all ATen operators from a model by tracing with TorchScript.
    
    Args:
        model: PyTorch model instance
        dummy_input: Dummy input tensor for tracing
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary containing operator information
    """
    operators = defaultdict(lambda: {
        "count": 0,
        "nodes": [],
    })
    
    try:
        model.eval()
        with torch.no_grad():
            if verbose:
                print("Tracing model with TorchScript...")
            traced_model = torch.jit.trace(model, dummy_input)
        
        graph = traced_model.graph
        
        def process_node(node, depth=0):
            """Recursively process nodes in the graph."""
            op_kind = node.kind()
            
            # Extract namespace and operator name
            if "::" in op_kind:
                namespace, op_name = op_kind.split("::", 1)
            else:
                namespace = "prim"
                op_name = op_kind
            
            # Only track ATen operators (actual PyTorch backend operations)
            if namespace == "aten":
                operators[op_name]["count"] += 1
                
                node_info = {
                    "kind": op_kind,
                    "inputs": [],
                    "outputs": [],
                }
                
                # Extract input information
                for inp in node.inputs():
                    inp_info = {}
                    if hasattr(inp, 'type'):
                        inp_info["type"] = str(inp.type())
                    if hasattr(inp, 'debugName') and inp.debugName():
                        inp_info["name"] = inp.debugName()
                    node_info["inputs"].append(inp_info)
                
                # Extract output information
                for out in node.outputs():
                    out_info = {}
                    if hasattr(out, 'type'):
                        out_info["type"] = str(out.type())
                    if hasattr(out, 'debugName') and out.debugName():
                        out_info["name"] = out.debugName()
                    node_info["outputs"].append(out_info)
                
                operators[op_name]["nodes"].append(node_info)
            
            # Process subgraphs (for control flow like if/loop)
            for block in node.blocks():
                for sub_node in block.nodes():
                    process_node(sub_node, depth + 1)
        
        # Process all nodes in the graph
        for node in graph.nodes():
            process_node(node)
        
        if verbose:
            print(f"Found {len(operators)} unique ATen operators")
            print(f"Total operator calls: {sum(op['count'] for op in operators.values())}")
    
    except Exception as e:
        print(f"Error during tracing: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    # Convert defaultdict to regular dict
    return dict(operators)


def save_operators_to_json(operators: Dict[str, Any], output_path: Path, 
                          model_info: Optional[Dict[str, Any]] = None):
    """Save operator information to JSON file."""
    result = {
        "operators": operators,
        "summary": {
            "total_unique_operators": len(operators),
            "total_operator_calls": sum(op["count"] for op in operators.values()),
        },
    }
    
    if model_info:
        result["model_info"] = model_info
    
    # Get top operators
    sorted_ops = sorted(operators.items(), key=lambda x: x[1]["count"], reverse=True)
    result["top_operators"] = [
        {"name": name, "count": info["count"]} 
        for name, info in sorted_ops[:20]
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ Saved operator information to {output_path}")
    print(f"  Total unique operators: {len(operators)}")
    print(f"  Total operator calls: {result['summary']['total_operator_calls']}")
    print(f"\n  Top 10 operators:")
    for op_info in result["top_operators"][:10]:
        print(f"    {op_info['name']}: {op_info['count']} calls")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract ATen operators from PyTorch model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from a model instance (requires Python script with model):
  python -c "
  from nest_ssl_project.tools.extract_operators import extract_operators_from_model, save_operators_to_json
  from pathlib import Path
  import torch
  
  # Load your model here
  model = ...  # Your model instance
  dummy_input = torch.randn(2, 80, 1584)  # Adjust shape as needed
  
  operators = extract_operators_from_model(model, dummy_input, verbose=True)
  save_operators_to_json(operators, Path('operators.json'))
  "
        """
    )
    parser.add_argument("--output", type=str, default="operators.json",
                       help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed information")
    
    args = parser.parse_args()
    
    print("""
This script requires a model instance to extract operators.
Please use it as a module or provide a model instance.

Example usage:
  from nest_ssl_project.tools.extract_operators import extract_operators_from_model, save_operators_to_json
  import torch
  from pathlib import Path
  
  # Your model instance
  model = ...
  dummy_input = torch.randn(2, 80, 1584)
  
  operators = extract_operators_from_model(model, dummy_input, verbose=True)
  save_operators_to_json(operators, Path('operators.json'))
    """)


if __name__ == '__main__':
    main()

