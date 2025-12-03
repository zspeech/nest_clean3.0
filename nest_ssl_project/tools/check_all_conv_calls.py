#!/usr/bin/env python3
"""
Check all calls to encoder.pre_encode.conv to identify clean vs noisy paths.
"""

import torch
import pickle
from pathlib import Path

def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK ALL CONV CALLS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get preprocessor outputs for comparison
    nemo_preproc = nemo_layers.get('preprocessor', {})
    nest_preproc = nest_layers.get('preprocessor', {})
    
    nemo_preproc_outs = nemo_preproc.get('all_forward_outputs', [])
    nest_preproc_outs = nest_preproc.get('all_forward_outputs', [])
    
    nemo_call0_out = None
    nemo_call1_out = None
    
    if len(nemo_preproc_outs) >= 1:
        nemo_call0_out = nemo_preproc_outs[0][0]
    if len(nemo_preproc_outs) >= 2:
        nemo_call1_out = nemo_preproc_outs[1][0]
        
    nest_call0_out = None
    nest_call1_out = None
    
    if len(nest_preproc_outs) >= 1:
        nest_call0_out = nest_preproc_outs[0][0]
    if len(nest_preproc_outs) >= 2:
        nest_call1_out = nest_preproc_outs[1][0]

    # Check encoder.pre_encode.conv calls
    nemo_conv = nemo_layers.get('encoder.pre_encode.conv', {})
    nest_conv = nest_layers.get('encoder.pre_encode.conv', {})
    
    nemo_conv_inputs = nemo_conv.get('all_forward_inputs', [])
    nest_conv_inputs = nest_conv.get('all_forward_inputs', [])
    
    print(f"\nNeMo conv calls: {len(nemo_conv_inputs)}")
    for i, inp in enumerate(nemo_conv_inputs):
        if isinstance(inp, (list, tuple)): inp = inp[0]
        if not isinstance(inp, torch.Tensor): continue
        
        print(f"  Call {i}: shape={inp.shape}, mean={inp.float().mean():.6f}")
        
        # Compare with preprocessor outputs (simulated transpose)
        if nemo_call0_out is not None:
            sim_call0 = nemo_call0_out.transpose(1, 2)
            if inp.shape == sim_call0.shape:
                diff = (inp.float() - sim_call0.float()).abs().max().item()
                print(f"    vs Preproc Call 0 (sim): max_diff={diff:.6e}")
            elif inp.shape == sim_call0.unsqueeze(1).shape:
                 diff = (inp.float() - sim_call0.unsqueeze(1).float()).abs().max().item()
                 print(f"    vs Preproc Call 0 (sim+unsqueezed): max_diff={diff:.6e}")

        if nemo_call1_out is not None:
            sim_call1 = nemo_call1_out.transpose(1, 2)
            if inp.shape == sim_call1.shape:
                diff = (inp.float() - sim_call1.float()).abs().max().item()
                print(f"    vs Preproc Call 1 (sim): max_diff={diff:.6e}")
            elif inp.shape == sim_call1.unsqueeze(1).shape:
                 diff = (inp.float() - sim_call1.unsqueeze(1).float()).abs().max().item()
                 print(f"    vs Preproc Call 1 (sim+unsqueezed): max_diff={diff:.6e}")

    print(f"\nnest conv calls: {len(nest_conv_inputs)}")
    for i, inp in enumerate(nest_conv_inputs):
        if isinstance(inp, (list, tuple)): inp = inp[0]
        if not isinstance(inp, torch.Tensor): continue
        
        print(f"  Call {i}: shape={inp.shape}, mean={inp.float().mean():.6f}")
        
        if nest_call0_out is not None:
            sim_call0 = nest_call0_out.transpose(1, 2)
            if inp.shape == sim_call0.shape:
                diff = (inp.float() - sim_call0.float()).abs().max().item()
                print(f"    vs Preproc Call 0 (sim): max_diff={diff:.6e}")
            elif inp.shape == sim_call0.unsqueeze(1).shape:
                 diff = (inp.float() - sim_call0.unsqueeze(1).float()).abs().max().item()
                 print(f"    vs Preproc Call 0 (sim+unsqueezed): max_diff={diff:.6e}")

        if nest_call1_out is not None:
            sim_call1 = nest_call1_out.transpose(1, 2)
            if inp.shape == sim_call1.shape:
                diff = (inp.float() - sim_call1.float()).abs().max().item()
                print(f"    vs Preproc Call 1 (sim): max_diff={diff:.6e}")
            elif inp.shape == sim_call1.unsqueeze(1).shape:
                 diff = (inp.float() - sim_call1.unsqueeze(1).float()).abs().max().item()
                 print(f"    vs Preproc Call 1 (sim+unsqueezed): max_diff={diff:.6e}")

if __name__ == '__main__':
    main()
