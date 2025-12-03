#!/usr/bin/env python3
"""
Check if spec_augmentation is running and modifying the signal.
"""

import torch
import pickle
from pathlib import Path

def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK SPEC AUGMENTATION")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Check for spec_augmentation in hooks
    nemo_spec_aug = nemo_layers.get('spec_augmentation', {})
    nest_spec_aug = nest_layers.get('spec_augmentation', {})
    
    print(f"NeMo spec_augmentation captured: {bool(nemo_spec_aug)}")
    print(f"nest spec_augmentation captured: {bool(nest_spec_aug)}")
    
    if nemo_spec_aug:
        inputs = nemo_spec_aug.get('all_forward_inputs', [])
        outputs = nemo_spec_aug.get('all_forward_outputs', [])
        print(f"NeMo spec_aug calls: {len(inputs)}")
        if len(inputs) > 0 and len(outputs) > 0:
            # Use the last call (noisy signal)
            inp = inputs[-1][0] if isinstance(inputs[-1], (list, tuple)) else inputs[-1]
            out = outputs[-1][0] if isinstance(outputs[-1], (list, tuple)) else outputs[-1]
            
            if isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor):
                diff = (inp.float() - out.float()).abs().max().item()
                print(f"NeMo spec_aug input vs output max diff: {diff:.6e}")
                if diff > 1e-6:
                    print("  [WARNING] NeMo spec_augmentation is modifying the signal!")
                else:
                    print("  [OK] NeMo spec_augmentation is identity.")

            # Compare spec_aug output with conv input
            nemo_conv = nemo_layers.get('encoder.pre_encode.conv', {})
            if nemo_conv:
                conv_inp = nemo_conv.get('all_forward_inputs', [])[0]
                if isinstance(conv_inp, (list, tuple)): conv_inp = conv_inp[0]
                
                # spec_aug output is [B, D, T], conv input is [B, 1, T, D] (NeMo) or [B, T, D]
                # Need to transpose and unsqueeze to compare
                spec_aug_out_transposed = out.transpose(1, 2) # [B, T, D]
                
                print(f"NeMo spec_aug output shape: {out.shape}")
                print(f"NeMo conv input shape: {conv_inp.shape}")
                
                if conv_inp.shape == spec_aug_out_transposed.unsqueeze(1).shape:
                     diff_conv = (spec_aug_out_transposed.unsqueeze(1).float() - conv_inp.float()).abs().max().item()
                     print(f"NeMo spec_aug output vs conv input max diff: {diff_conv:.6e}")
                elif conv_inp.shape == spec_aug_out_transposed.shape: # if no unsqueeze
                     diff_conv = (spec_aug_out_transposed.float() - conv_inp.float()).abs().max().item()
                     print(f"NeMo spec_aug output vs conv input max diff: {diff_conv:.6e}")

    if nest_spec_aug:
        inputs = nest_spec_aug.get('all_forward_inputs', [])
        outputs = nest_spec_aug.get('all_forward_outputs', [])
        print(f"nest spec_aug calls: {len(inputs)}")
        if len(inputs) > 0 and len(outputs) > 0:
             # Use the last call
            inp = inputs[-1][0] if isinstance(inputs[-1], (list, tuple)) else inputs[-1]
            out = outputs[-1][0] if isinstance(outputs[-1], (list, tuple)) else outputs[-1]
            
            if isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor):
                diff = (inp.float() - out.float()).abs().max().item()
                print(f"nest spec_aug input vs output max diff: {diff:.6e}")
                if diff > 1e-6:
                    print("  [WARNING] nest spec_augmentation is modifying the signal!")
                else:
                    print("  [OK] nest spec_augmentation is identity.")

            # Compare spec_aug output with conv input
            nest_conv = nest_layers.get('encoder.pre_encode.conv', {})
            if nest_conv:
                conv_inp = nest_conv.get('all_forward_inputs', [])[0]
                if isinstance(conv_inp, (list, tuple)): conv_inp = conv_inp[0]
                
                spec_aug_out_transposed = out.transpose(1, 2) # [B, T, D]
                
                print(f"nest spec_aug output shape: {out.shape}")
                print(f"nest conv input shape: {conv_inp.shape}")

                if conv_inp.shape == spec_aug_out_transposed.shape:
                     diff_conv = (spec_aug_out_transposed.float() - conv_inp.float()).abs().max().item()
                     print(f"nest spec_aug output vs conv input max diff: {diff_conv:.6e}")

if __name__ == '__main__':
    main()

