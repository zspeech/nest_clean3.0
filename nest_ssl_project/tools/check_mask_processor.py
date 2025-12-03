#!/usr/bin/env python3
"""
Check mask_processor output and compare with encoder.pre_encode.conv input.
"""

import torch
import pickle
from pathlib import Path

def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK MASK PROCESSOR")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Check mask_processor
    nemo_mask_proc = nemo_layers.get('mask_processor', {})
    nest_mask_proc = nest_layers.get('mask_processor', {})
    
    nemo_mask_outs = nemo_mask_proc.get('all_forward_outputs', [])
    nest_mask_outs = nest_mask_proc.get('all_forward_outputs', [])
    
    nemo_masked_signal = None
    nest_masked_signal = None
    
    if len(nemo_mask_outs) > 0:
        # Output is (masked_feats, masks)
        nemo_out = nemo_mask_outs[0]
        if isinstance(nemo_out, (list, tuple)):
            nemo_masked_signal = nemo_out[0]
            print(f"NeMo mask_processor output[0] shape: {nemo_masked_signal.shape}")
            print(f"NeMo mask_processor output[0] mean: {nemo_masked_signal.float().mean():.6f}")
    
    if len(nest_mask_outs) > 0:
        nest_out = nest_mask_outs[0]
        if isinstance(nest_out, (list, tuple)):
            nest_masked_signal = nest_out[0]
            print(f"nest mask_processor output[0] shape: {nest_masked_signal.shape}")
            print(f"nest mask_processor output[0] mean: {nest_masked_signal.float().mean():.6f}")

    # Compare mask_processor output with conv input
    nemo_conv = nemo_layers.get('encoder.pre_encode.conv', {})
    nest_conv = nest_layers.get('encoder.pre_encode.conv', {})
    
    if nemo_conv and nemo_masked_signal is not None:
        nemo_conv_input = nemo_conv.get('all_forward_inputs', [])[0]
        if isinstance(nemo_conv_input, (list, tuple)): nemo_conv_input = nemo_conv_input[0]
        
        # mask_processor output is [B, D, T]. conv input is [B, 1, T, D] (NeMo) or [B, T, D]
        # Need to transpose and unsqueeze
        nemo_sim_input = nemo_masked_signal.transpose(1, 2) # [B, T, D]
        
        print(f"NeMo conv input shape: {nemo_conv_input.shape}")
        
        if nemo_conv_input.shape == nemo_sim_input.shape:
            diff = (nemo_conv_input.float() - nemo_sim_input.float()).abs().max().item()
            print(f"NeMo mask_processor (sim) vs conv input max diff: {diff:.6e}")
        elif nemo_conv_input.shape == nemo_sim_input.unsqueeze(1).shape:
            diff = (nemo_conv_input.float() - nemo_sim_input.unsqueeze(1).float()).abs().max().item()
            print(f"NeMo mask_processor (sim+unsqueezed) vs conv input max diff: {diff:.6e}")

    if nest_conv and nest_masked_signal is not None:
        nest_conv_input = nest_conv.get('all_forward_inputs', [])[0]
        if isinstance(nest_conv_input, (list, tuple)): nest_conv_input = nest_conv_input[0]
        
        nest_sim_input = nest_masked_signal.transpose(1, 2) # [B, T, D]
        
        print(f"nest conv input shape: {nest_conv_input.shape}")
        
        if nest_conv_input.shape == nest_sim_input.shape:
            diff = (nest_conv_input.float() - nest_sim_input.float()).abs().max().item()
            print(f"nest mask_processor (sim) vs conv input max diff: {diff:.6e}")
        elif nest_conv_input.shape == nest_sim_input.unsqueeze(1).shape:
            diff = (nest_conv_input.float() - nest_sim_input.unsqueeze(1).float()).abs().max().item()
            print(f"nest mask_processor (sim+unsqueezed) vs conv input max diff: {diff:.6e}")

if __name__ == '__main__':
    main()

