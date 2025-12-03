#!/usr/bin/env python3
"""
Trace MaskedConvSequential step by step to find where the mismatch occurs.
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
    print("TRACE MASKED CONV SEQUENTIAL STEP BY STEP")
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
    
    # Step 1: transpose(1, 2) in ConformerEncoder.forward_internal
    nemo_x = nemo_preproc_call1.transpose(1, 2)  # [B, T, D]
    nest_x = nest_preproc_call1.transpose(1, 2)  # [B, T, D]
    
    print(f"\nStep 1: After transpose(1, 2)")
    print(f"  NeMo: shape={nemo_x.shape}, mean={nemo_x.float().mean():.6f}")
    print(f"  nest: shape={nest_x.shape}, mean={nest_x.float().mean():.6f}")
    
    diff_step1 = (nemo_x.float() - nest_x.float()).abs()
    max_diff_step1 = diff_step1.max().item()
    print(f"  Comparison: max_diff={max_diff_step1:.6e} {'[OK]' if max_diff_step1 < 1e-5 else '[FAIL]'}")
    
    # Step 2: unsqueeze(1) in MaskedConvSequential.forward
    nemo_x_unsqueezed = nemo_x.unsqueeze(1)  # [B, 1, T, D]
    nest_x_unsqueezed = nest_x.unsqueeze(1)  # [B, 1, T, D]
    
    print(f"\nStep 2: After unsqueeze(1)")
    print(f"  NeMo: shape={nemo_x_unsqueezed.shape}, mean={nemo_x_unsqueezed.float().mean():.6f}")
    print(f"  nest: shape={nest_x_unsqueezed.shape}, mean={nest_x_unsqueezed.float().mean():.6f}")
    
    diff_step2 = (nemo_x_unsqueezed.float() - nest_x_unsqueezed.float()).abs()
    max_diff_step2 = diff_step2.max().item()
    print(f"  Comparison: max_diff={max_diff_step2:.6e} {'[OK]' if max_diff_step2 < 1e-5 else '[FAIL]'}")
    
    # Step 3: Create mask
    nemo_mask = _create_mask(nemo_x_unsqueezed, nemo_seq_len)
    nest_mask = _create_mask(nest_x_unsqueezed, nest_seq_len)
    
    print(f"\nStep 3: Create mask")
    print(f"  NeMo: shape={nemo_mask.shape}, mean={nemo_mask.float().mean():.6f}")
    print(f"  nest: shape={nest_mask.shape}, mean={nest_mask.float().mean():.6f}")
    
    diff_step3 = (nemo_mask.float() - nest_mask.float()).abs()
    max_diff_step3 = diff_step3.max().item()
    print(f"  Comparison: max_diff={max_diff_step3:.6e} {'[OK]' if max_diff_step3 < 1e-5 else '[FAIL]'}")
    
    # Step 4: Apply mask before first layer
    nemo_x_masked = apply_channel_mask(nemo_x_unsqueezed, nemo_mask)
    nest_x_masked = apply_channel_mask(nest_x_unsqueezed, nest_mask)
    
    print(f"\nStep 4: After mask application (before first layer)")
    print(f"  NeMo: shape={nemo_x_masked.shape}, mean={nemo_x_masked.float().mean():.6f}")
    print(f"  nest: shape={nest_x_masked.shape}, mean={nest_x_masked.float().mean():.6f}")
    
    diff_step4 = (nemo_x_masked.float() - nest_x_masked.float()).abs()
    max_diff_step4 = diff_step4.max().item()
    mean_diff_step4 = diff_step4.mean().item()
    print(f"  Comparison: max_diff={max_diff_step4:.6e}, mean_diff={mean_diff_step4:.6e} {'[OK]' if max_diff_step4 < 1e-5 else '[FAIL]'}")
    
    # Step 5: Actual conv.0 input
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
            print(f"\nStep 5: Actual conv.0 input (from hook)")
            print(f"  NeMo: shape={nemo_conv0_input_tensor.shape}, mean={nemo_conv0_input_tensor.float().mean():.6f}")
            print(f"  nest: shape={nest_conv0_input_tensor.shape}, mean={nest_conv0_input_tensor.float().mean():.6f}")
            
            # Compare Step 4 vs Step 5
            diff_nemo_step4_vs_step5 = (nemo_x_masked.float() - nemo_conv0_input_tensor.float()).abs()
            max_diff_nemo_step4_vs_step5 = diff_nemo_step4_vs_step5.max().item()
            mean_diff_nemo_step4_vs_step5 = diff_nemo_step4_vs_step5.mean().item()
            print(f"\nNeMo Step 4 vs Step 5:")
            print(f"  max_diff={max_diff_nemo_step4_vs_step5:.6e}, mean_diff={mean_diff_nemo_step4_vs_step5:.6e} {'[OK]' if max_diff_nemo_step4_vs_step5 < 1e-5 else '[FAIL]'}")
            
            diff_nest_step4_vs_step5 = (nest_x_masked.float() - nest_conv0_input_tensor.float()).abs()
            max_diff_nest_step4_vs_step5 = diff_nest_step4_vs_step5.max().item()
            mean_diff_nest_step4_vs_step5 = diff_nest_step4_vs_step5.mean().item()
            print(f"\nnest Step 4 vs Step 5:")
            print(f"  max_diff={max_diff_nest_step4_vs_step5:.6e}, mean_diff={mean_diff_nest_step4_vs_step5:.6e} {'[OK]' if max_diff_nest_step4_vs_step5 < 1e-5 else '[FAIL]'}")
            
            # Compare NeMo vs nest Step 5
            diff_step5 = (nemo_conv0_input_tensor.float() - nest_conv0_input_tensor.float()).abs()
            max_diff_step5 = diff_step5.max().item()
            mean_diff_step5 = diff_step5.mean().item()
            print(f"\nNeMo vs nest Step 5:")
            print(f"  max_diff={max_diff_step5:.6e}, mean_diff={mean_diff_step5:.6e} {'[OK]' if max_diff_step5 < 1e-5 else '[FAIL]'}")


if __name__ == '__main__':
    main()

