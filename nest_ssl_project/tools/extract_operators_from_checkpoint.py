#!/usr/bin/env python3
"""
Extract actual PyTorch ATen operators from a saved model checkpoint.
This script loads the model and traces it to extract all operator calls.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any
import sys


def extract_operators_from_model(model: nn.Module, dummy_input: torch.Tensor) -> Dict[str, Any]:
    """Extract all ATen operators from a model by tracing with TorchScript."""
    operators = defaultdict(lambda: {
        "count": 0,
        "nodes": [],
    })
    
    try:
        model.eval()
        with torch.no_grad():
            print("Tracing model with TorchScript...")
            traced_model = torch.jit.trace(model, dummy_input)
        
        graph = traced_model.graph
        
        def process_node(node):
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
            
            # Process subgraphs (for control flow)
            for block in node.blocks():
                for sub_node in block.nodes():
                    process_node(sub_node)
        
        # Process all nodes in the graph
        for node in graph.nodes():
            process_node(node)
        
        print(f"Found {len(operators)} unique ATen operators")
        print(f"Total operator calls: {sum(op['count'] for op in operators.values())}")
    
    except Exception as e:
        print(f"Error during tracing: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    return dict(operators)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ATen operators from model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pt or .pth file)")
    parser.add_argument("--model_class", type=str, required=True,
                       help="Model class name (e.g., EncDecDenoiseMaskedTokenPredModel)")
    parser.add_argument("--config_path", type=str,
                       help="Path to config file (if needed)")
    parser.add_argument("--dummy_input_shape", type=str, default="2,80,1584",
                       help="Dummy input shape (comma-separated, e.g., '2,80,1584')")
    parser.add_argument("--output", type=str, default="operators.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    # Parse dummy input shape
    shape_parts = [int(x) for x in args.dummy_input_shape.split(',')]
    dummy_input = torch.randn(*shape_parts)
    print(f"Using dummy input shape: {dummy_input.shape}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Try to instantiate model
    # This is a simplified version - you may need to adapt based on your model structure
    print(f"Attempting to instantiate model class: {args.model_class}")
    
    # Import model class
    try:
        # Try common import paths
        if args.model_class == "EncDecDenoiseMaskedTokenPredModel":
            try:
                from models.ssl_models import EncDecDenoiseMaskedTokenPredModel as ModelClass
            except ImportError:
                try:
                    from nest_ssl_project.models.ssl_models import EncDecDenoiseMaskedTokenPredModel as ModelClass
                except ImportError:
                    print("Error: Could not import model class. Please ensure the model is importable.")
                    print("You may need to add the project root to PYTHONPATH.")
                    return
        
        # Instantiate model (simplified - you may need to provide config)
        print("Warning: Model instantiation requires config. Please provide a working model instance.")
        print("For now, this script serves as a template.")
        print("\nTo use this script:")
        print("1. Import the model class in your Python environment")
        print("2. Load the checkpoint and instantiate the model")
        print("3. Call extract_operators_from_model(model, dummy_input)")
        print("\nExample:")
        print("""
from nest_ssl_project.tools.extract_operators_from_checkpoint import extract_operators_from_model
import torch
import json
from pathlib import Path

# Load your model
model = ...  # Your model instance
dummy_input = torch.randn(2, 80, 1584)

# Extract operators
operators = extract_operators_from_model(model, dummy_input)

# Save to JSON
with open('operators.json', 'w') as f:
    json.dump(operators, f, indent=2, default=str)
        """)
        return
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()

