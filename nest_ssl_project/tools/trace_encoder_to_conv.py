#!/usr/bin/env python3
"""
Trace data flow from encoder input to pre_encode.conv input.
"""

import torch
import pickle
from pathlib import Path
import argparse


def load_layer_outputs(output_dir):
    """Load layer outputs from saved files."""
    step_dir = Path(output_dir) / "step_0"
    
    with open(step_dir / 'layer_outputs.pkl', 'rb') as f:
        layer_outputs = pickle.load(f)
    
    return layer_outputs


def compare_tensors(t1, t2, name=""):
    """Compare two tensors."""
    if t1 is None or t2 is None:
        print(f"  {name}: One is None")
        return
    
    if isinstance(t1, (list, tuple)):
        t1 = t1[0] if len(t1) > 0 else None
    if isinstance(t2, (list, tuple)):
        t2 = t2[0] if len(t2) > 0 else None
    
    if t1 is None or t2 is None:
        print(f"  {name}: One is None after unpacking")
        return
    
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        print(f"  {name}: Not tensors")
        return
    
    diff = (t1.float() - t2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  {name}:")
    print(f"    Shape: {t1.shape}")
    print(f"    max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    print(f"    NeMo mean={t1.float().mean().item():.6f}, nest mean={t2.float().mean().item():.6f}")
    
    if max_diff > 1e-5:
        print(f"    [FAIL]")
    else:
        print(f"    [OK]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_dir', type=str, default='./saved_nemo_outputs')
    parser.add_argument('--nest_dir', type=str, default='./saved_nest_outputs')
    args = parser.parse_args()
    
    print("="*80)
    print("ENCODER TO CONV TRACE")
    print("="*80)
    
    # Load layer outputs
    nemo_layers = load_layer_outputs(args.nemo_dir)
    nest_layers = load_layer_outputs(args.nest_dir)
    
    # Load preprocessor outputs (Call 1 - noisy signal)
    nemo_preprocessor_outputs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preprocessor_outputs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    
    if len(nemo_preprocessor_outputs) < 2 or len(nest_preprocessor_outputs) < 2:
        print("ERROR: Not enough preprocessor calls captured")
        return
    
    nemo_preprocessor_out = nemo_preprocessor_outputs[1][0]  # Call 1, output[0]
    nest_preprocessor_out = nest_preprocessor_outputs[1][0]  # Call 1, output[0]
    
    print("\n1. Preprocessor output (Call 1 - noisy signal)")
    compare_tensors(nemo_preprocessor_out, nest_preprocessor_out, "preprocessor_out")
    
    # Get encoder input
    nemo_encoder_input = nemo_layers.get('encoder', {}).get('forward_inputs')
    nest_encoder_input = nest_layers.get('encoder', {}).get('forward_inputs')
    
    print("\n2. Encoder forward_inputs (from hook)")
    if nemo_encoder_input is not None and nest_encoder_input is not None:
        if isinstance(nemo_encoder_input, (list, tuple)) and len(nemo_encoder_input) > 0:
            compare_tensors(nemo_encoder_input[0], nest_encoder_input[0], "encoder_input[0]")
        else:
            print("  Encoder input format unexpected")
    else:
        print("  Encoder input not captured")
    
    # Simulate transpose: [B, D, T] -> [B, T, D]
    print("\n3. After transpose(1, 2)")
    nemo_transposed = nemo_preprocessor_out.transpose(1, 2)
    nest_transposed = nest_preprocessor_out.transpose(1, 2)
    compare_tensors(nemo_transposed, nest_transposed, "transposed")
    
    # Get pre_encode input
    nemo_pre_encode_input = nemo_layers.get('encoder.pre_encode', {}).get('forward_inputs')
    nest_pre_encode_input = nest_layers.get('encoder.pre_encode', {}).get('forward_inputs')
    
    print("\n4. pre_encode forward_inputs (from hook)")
    if nemo_pre_encode_input is not None and nest_pre_encode_input is not None:
        if isinstance(nemo_pre_encode_input, (list, tuple)) and len(nemo_pre_encode_input) > 0:
            compare_tensors(nemo_pre_encode_input[0], nest_pre_encode_input[0], "pre_encode_input[0]")
        else:
            print("  pre_encode input format unexpected")
    else:
        print("  pre_encode input not captured")
    
    # Get pre_encode.conv input
    nemo_conv_input = nemo_layers.get('encoder.pre_encode.conv', {}).get('forward_inputs')
    nest_conv_input = nest_layers.get('encoder.pre_encode.conv', {}).get('forward_inputs')
    
    print("\n5. pre_encode.conv forward_inputs (from hook)")
    if nemo_conv_input is not None and nest_conv_input is not None:
        if isinstance(nemo_conv_input, (list, tuple)) and len(nemo_conv_input) > 0:
            compare_tensors(nemo_conv_input[0], nest_conv_input[0], "conv_input[0]")
        else:
            print("  conv input format unexpected")
    else:
        print("  conv input not captured")
    
    # Simulate unsqueeze(1): [B, T, D] -> [B, 1, T, D]
    print("\n6. Simulated unsqueeze(1) from transposed")
    nemo_unsqueezed = nemo_transposed.unsqueeze(1)
    nest_unsqueezed = nest_transposed.unsqueeze(1)
    compare_tensors(nemo_unsqueezed, nest_unsqueezed, "unsqueezed")
    
    # Compare simulated vs actual conv input
    print("\n7. Simulated vs actual conv input")
    if nemo_conv_input is not None and nest_conv_input is not None:
        if isinstance(nemo_conv_input, (list, tuple)) and len(nemo_conv_input) > 0:
            nemo_conv_actual = nemo_conv_input[0]
            nest_conv_actual = nest_conv_input[0]
            
            compare_tensors(nemo_unsqueezed, nemo_conv_actual, "NeMo: simulated vs actual")
            compare_tensors(nest_unsqueezed, nest_conv_actual, "nest: simulated vs actual")


if __name__ == '__main__':
    main()

