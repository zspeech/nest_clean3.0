#!/usr/bin/env python3
"""
Check mask creation and application in MaskedConvSequential.
"""

import torch
import pickle
from pathlib import Path


def _create_mask(tensor, lengths):
    """Create mask matching tensor dimensions."""
    batch_size, channels, time, features = tensor.shape
    time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
    return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)


def apply_channel_mask(tensor, mask):
    """Apply mask to tensor with channel dimension."""
    batch_size, channels, time, features = tensor.shape
    expanded_mask = mask.unsqueeze(1).expand(batch_size, channels, time, features)
    return tensor * expanded_mask


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK MASK CREATION AND APPLICATION")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get preprocessor Call 1 output
    nemo_preproc_outs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preproc_outs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    
    nemo_preproc_call1 = nemo_preproc_outs[1][0]  # Call 1, output[0]
    nest_preproc_call1 = nest_preproc_outs[1][0]  # Call 1, output[0]
    
    nemo_seq_len = nemo_preproc_outs[1][1]  # Call 1, output[1] (seq_len)
    nest_seq_len = nest_preproc_outs[1][1]  # Call 1, output[1] (seq_len)
    
    # Simulate: transpose then unsqueeze
    nemo_x = nemo_preproc_call1.transpose(1, 2).unsqueeze(1)  # [B, 1, T, D]
    nest_x = nest_preproc_call1.transpose(1, 2).unsqueeze(1)  # [B, 1, T, D]
    
    print(f"\nAfter transpose + unsqueeze:")
    print(f"  NeMo: shape={nemo_x.shape}, mean={nemo_x.float().mean():.6f}")
    print(f"  nest: shape={nest_x.shape}, mean={nest_x.float().mean():.6f}")
    
    # Compare
    diff_x = (nemo_x.float() - nest_x.float()).abs()
    max_diff_x = diff_x.max().item()
    print(f"  Comparison: max_diff={max_diff_x:.6e} {'[OK]' if max_diff_x < 1e-5 else '[FAIL]'}")
    
    # Create mask
    nemo_mask = _create_mask(nemo_x, nemo_seq_len)
    nest_mask = _create_mask(nest_x, nest_seq_len)
    
    print(f"\nMask:")
    print(f"  NeMo: shape={nemo_mask.shape}, mean={nemo_mask.float().mean():.6f}")
    print(f"  nest: shape={nest_mask.shape}, mean={nest_mask.float().mean():.6f}")
    
    # Compare mask
    diff_mask = (nemo_mask.float() - nest_mask.float()).abs()
    max_diff_mask = diff_mask.max().item()
    print(f"  Comparison: max_diff={max_diff_mask:.6e} {'[OK]' if max_diff_mask < 1e-5 else '[FAIL]'}")
    
    # Apply mask
    nemo_masked = apply_channel_mask(nemo_x, nemo_mask)
    nest_masked = apply_channel_mask(nest_x, nest_mask)
    
    print(f"\nAfter mask application:")
    print(f"  NeMo: shape={nemo_masked.shape}, mean={nemo_masked.float().mean():.6f}")
    print(f"  nest: shape={nest_masked.shape}, mean={nest_masked.float().mean():.6f}")
    
    # Compare masked
    diff_masked = (nemo_masked.float() - nest_masked.float()).abs()
    max_diff_masked = diff_masked.max().item()
    mean_diff_masked = diff_masked.mean().item()
    print(f"  Comparison: max_diff={max_diff_masked:.6e}, mean_diff={mean_diff_masked:.6e} {'[OK]' if max_diff_masked < 1e-5 else '[FAIL]'}")
    
    # Compare with actual conv.0 input
    nemo_conv0 = nemo_layers.get('encoder.pre_encode.conv.0', {})
    nest_conv0 = nest_layers.get('encoder.pre_encode.conv.0', {})
    
    nemo_conv0_inputs = nemo_conv0.get('all_forward_inputs', [])
    nest_conv0_inputs = nest_conv0.get('all_forward_inputs', [])
    
    if len(nemo_conv0_inputs) > 0 and len(nest_conv0_inputs) > 0:
        nemo_conv0_input = nemo_conv0_inputs[0]
        nest_conv0_input = nest_conv0_inputs[0]
        
        if isinstance(nemo_conv0_input, (list, tuple)) and len(nemo_conv0_input) > 0:
            nemo_conv0_input_tensor = nemo_conv0_input[0]
        else:
            nemo_conv0_input_tensor = nemo_conv0_input
        
        if isinstance(nest_conv0_input, (list, tuple)) and len(nest_conv0_input) > 0:
            nest_conv0_input_tensor = nest_conv0_input[0]
        else:
            nest_conv0_input_tensor = nest_conv0_input
        
        if isinstance(nemo_conv0_input_tensor, torch.Tensor) and isinstance(nest_conv0_input_tensor, torch.Tensor):
            print(f"\nActual conv.0 input:")
            print(f"  NeMo: shape={nemo_conv0_input_tensor.shape}, mean={nemo_conv0_input_tensor.float().mean():.6f}")
            print(f"  nest: shape={nest_conv0_input_tensor.shape}, mean={nest_conv0_input_tensor.float().mean():.6f}")
            
            # Compare simulated vs actual
            diff_nemo_sim = (nemo_masked.float() - nemo_conv0_input_tensor.float()).abs()
            max_diff_nemo_sim = diff_nemo_sim.max().item()
            print(f"\nNeMo: simulated vs actual:")
            print(f"  max_diff={max_diff_nemo_sim:.6e} {'[OK]' if max_diff_nemo_sim < 1e-5 else '[FAIL]'}")
            
            diff_nest_sim = (nest_masked.float() - nest_conv0_input_tensor.float()).abs()
            max_diff_nest_sim = diff_nest_sim.max().item()
            print(f"nest: simulated vs actual:")
            print(f"  max_diff={max_diff_nest_sim:.6e} {'[OK]' if max_diff_nest_sim < 1e-5 else '[FAIL]'}")


if __name__ == '__main__':
    main()

