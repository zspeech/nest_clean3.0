#!/usr/bin/env python3
"""
Trace the source of encoder input and compare with preprocessor outputs.
"""

import torch
import pickle
from pathlib import Path
import argparse
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add paths for pickle deserialization
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'NeMo'))
sys.path.insert(0, str(project_root / 'nest_ssl_project'))


def load_layer_outputs(output_dir):
    """Load layer outputs from saved files."""
    step_dir = Path(output_dir) / "step_0"
    
    try:
        with open(step_dir / 'layer_outputs.pkl', 'rb') as f:
            layer_outputs = pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load layer_outputs.pkl: {e}")
        print("Trying with weights_only=False...")
        with open(step_dir / 'layer_outputs.pkl', 'rb') as f:
            import pickle
            layer_outputs = pickle.load(f)
    
    return layer_outputs


def compare_tensors(t1, t2, name=""):
    """Compare two tensors and return detailed info."""
    if t1 is None or t2 is None:
        return f"{name}: One is None"
    
    if isinstance(t1, (list, tuple)):
        t1 = t1[0] if len(t1) > 0 else None
    if isinstance(t2, (list, tuple)):
        t2 = t2[0] if len(t2) > 0 else None
    
    if t1 is None or t2 is None:
        return f"{name}: One is None after unpacking"
    
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        return f"{name}: Not tensors"
    
    diff = (t1.float() - t2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    status = "[OK]" if max_diff < 1e-5 else "[FAIL]"
    return f"{name}: {status} max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_dir', type=str, default='./saved_nemo_outputs')
    parser.add_argument('--nest_dir', type=str, default='./saved_nest_outputs')
    args = parser.parse_args()
    
    print("="*80)
    print("ENCODER INPUT SOURCE TRACE")
    print("="*80)
    
    # Load layer outputs
    nemo_layers = load_layer_outputs(args.nemo_dir)
    nest_layers = load_layer_outputs(args.nest_dir)
    
    # Get all preprocessor outputs
    nemo_preprocessor_outputs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preprocessor_outputs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    
    print(f"\nNeMo preprocessor calls: {len(nemo_preprocessor_outputs)}")
    print(f"nest preprocessor calls: {len(nest_preprocessor_outputs)}")
    
    # Get encoder inputs
    nemo_encoder_inputs = nemo_layers.get('encoder', {}).get('all_forward_inputs', [])
    nest_encoder_inputs = nest_layers.get('encoder', {}).get('all_forward_inputs', [])
    
    print(f"\nNeMo encoder calls: {len(nemo_encoder_inputs)}")
    print(f"nest encoder calls: {len(nest_encoder_inputs)}")
    
    # Compare encoder input with each preprocessor output
    if len(nemo_encoder_inputs) > 0 and len(nemo_preprocessor_outputs) > 0:
        nemo_encoder_input = nemo_encoder_inputs[0][0] if isinstance(nemo_encoder_inputs[0], (list, tuple)) else nemo_encoder_inputs[0]
        
        print("\n" + "="*80)
        print("NeMo: Compare encoder input with preprocessor outputs")
        print("="*80)
        
        for i, preproc_out in enumerate(nemo_preprocessor_outputs):
            preproc_out_tensor = preproc_out[0] if isinstance(preproc_out, (list, tuple)) else preproc_out
            print(f"\nCall {i}: {compare_tensors(nemo_encoder_input, preproc_out_tensor, f'encoder_input vs preprocessor_output[{i}]')}")
            if isinstance(preproc_out_tensor, torch.Tensor):
                print(f"  preprocessor_output[{i}] shape: {preproc_out_tensor.shape}, mean: {preproc_out_tensor.float().mean():.6f}")
        
        print(f"\nencoder_input shape: {nemo_encoder_input.shape}, mean: {nemo_encoder_input.float().mean():.6f}")
    
    if len(nest_encoder_inputs) > 0 and len(nest_preprocessor_outputs) > 0:
        nest_encoder_input = nest_encoder_inputs[0][0] if isinstance(nest_encoder_inputs[0], (list, tuple)) else nest_encoder_inputs[0]
        
        print("\n" + "="*80)
        print("nest: Compare encoder input with preprocessor outputs")
        print("="*80)
        
        for i, preproc_out in enumerate(nest_preprocessor_outputs):
            preproc_out_tensor = preproc_out[0] if isinstance(preproc_out, (list, tuple)) else preproc_out
            print(f"\nCall {i}: {compare_tensors(nest_encoder_input, preproc_out_tensor, f'encoder_input vs preprocessor_output[{i}]')}")
            if isinstance(preproc_out_tensor, torch.Tensor):
                print(f"  preprocessor_output[{i}] shape: {preproc_out_tensor.shape}, mean: {preproc_out_tensor.float().mean():.6f}")
        
        print(f"\nencoder_input shape: {nest_encoder_input.shape}, mean: {nest_encoder_input.float().mean():.6f}")
    
    # Compare NeMo vs nest encoder inputs
    if len(nemo_encoder_inputs) > 0 and len(nest_encoder_inputs) > 0:
        nemo_encoder_input = nemo_encoder_inputs[0][0] if isinstance(nemo_encoder_inputs[0], (list, tuple)) else nemo_encoder_inputs[0]
        nest_encoder_input = nest_encoder_inputs[0][0] if isinstance(nest_encoder_inputs[0], (list, tuple)) else nest_encoder_inputs[0]
        
        print("\n" + "="*80)
        print("NeMo vs nest encoder inputs")
        print("="*80)
        print(f"\n{compare_tensors(nemo_encoder_input, nest_encoder_input, 'NeMo vs nest')}")


if __name__ == '__main__':
    main()

