#!/usr/bin/env python3
"""
Trace preprocessor calls and check which one is captured by hooks
"""

import torch
import pickle
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_dir", type=str, required=True)
    parser.add_argument("--nest_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir)
    nest_dir = Path(args.nest_dir)
    
    if args.step is None:
        nemo_steps = [int(d.name.split('_')[1]) for d in nemo_dir.glob("step_*") if d.is_dir()]
        nest_steps = [int(d.name.split('_')[1]) for d in nest_dir.glob("step_*") if d.is_dir()]
        common_steps = sorted(list(set(nemo_steps) & set(nest_steps)))
        if common_steps:
            args.step = common_steps[0]
    
    print("="*80)
    print("PREPROCESSOR CALLS TRACE")
    print("="*80)
    
    # Load batch data
    nemo_batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", map_location='cpu', weights_only=False)
    nest_batch = torch.load(nest_dir / f"step_{args.step}" / "batch.pt", map_location='cpu', weights_only=False)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get preprocessor inputs and outputs
    nemo_prep_inputs = nemo_layers.get('preprocessor', {}).get('forward_inputs')
    nest_prep_inputs = nest_layers.get('preprocessor', {}).get('forward_inputs')
    
    nemo_prep_outputs = nemo_layers.get('preprocessor', {}).get('forward_outputs')
    nest_prep_outputs = nest_layers.get('preprocessor', {}).get('forward_outputs')
    
    print("\n" + "="*60)
    print("1. Preprocessor Forward Inputs (from hooks)")
    print("="*60)
    
    if nemo_prep_inputs:
        print(f"NeMo preprocessor inputs: {len(nemo_prep_inputs)} items")
        for i, inp in enumerate(nemo_prep_inputs):
            if isinstance(inp, torch.Tensor):
                mean_val = inp.float().mean().item() if inp.numel() > 0 else 0
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}, mean={mean_val:.6f}")
    else:
        print("NeMo preprocessor inputs: None")
    
    if nest_prep_inputs:
        print(f"nest preprocessor inputs: {len(nest_prep_inputs)} items")
        for i, inp in enumerate(nest_prep_inputs):
            if isinstance(inp, torch.Tensor):
                mean_val = inp.float().mean().item() if inp.numel() > 0 else 0
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}, mean={mean_val:.6f}")
    else:
        print("nest preprocessor inputs: None")
    
    # Check if preprocessor input matches audio or noisy_audio
    print("\n" + "="*60)
    print("2. Check which audio the preprocessor received")
    print("="*60)
    
    nemo_audio = nemo_batch.get('audio')
    nemo_noisy = nemo_batch.get('noisy_audio')
    nest_audio = nest_batch.get('audio')
    nest_noisy = nest_batch.get('noisy_audio')
    
    if nemo_prep_inputs and len(nemo_prep_inputs) > 0:
        nemo_input = nemo_prep_inputs[0]
        if isinstance(nemo_input, torch.Tensor):
            if nemo_audio is not None and nemo_input.shape == nemo_audio.shape:
                diff_audio = (nemo_input - nemo_audio).abs().max().item()
                print(f"NeMo preprocessor input vs audio: max_diff={diff_audio:.6e}")
            if nemo_noisy is not None and nemo_input.shape == nemo_noisy.shape:
                diff_noisy = (nemo_input - nemo_noisy).abs().max().item()
                print(f"NeMo preprocessor input vs noisy_audio: max_diff={diff_noisy:.6e}")
    
    if nest_prep_inputs and len(nest_prep_inputs) > 0:
        nest_input = nest_prep_inputs[0]
        if isinstance(nest_input, torch.Tensor):
            if nest_audio is not None and nest_input.shape == nest_audio.shape:
                diff_audio = (nest_input - nest_audio).abs().max().item()
                print(f"nest preprocessor input vs audio: max_diff={diff_audio:.6e}")
            if nest_noisy is not None and nest_input.shape == nest_noisy.shape:
                diff_noisy = (nest_input - nest_noisy).abs().max().item()
                print(f"nest preprocessor input vs noisy_audio: max_diff={diff_noisy:.6e}")
    
    # Check featurizer inputs
    print("\n" + "="*60)
    print("3. Featurizer Forward Inputs")
    print("="*60)
    
    nemo_feat_inputs = nemo_layers.get('preprocessor.featurizer', {}).get('forward_inputs')
    nest_feat_inputs = nest_layers.get('preprocessor.featurizer', {}).get('forward_inputs')
    
    if nemo_feat_inputs:
        print(f"NeMo featurizer inputs: {len(nemo_feat_inputs)} items")
        for i, inp in enumerate(nemo_feat_inputs):
            if isinstance(inp, torch.Tensor):
                mean_val = inp.float().mean().item() if inp.numel() > 0 else 0
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}, mean={mean_val:.6f}")
    
    if nest_feat_inputs:
        print(f"nest featurizer inputs: {len(nest_feat_inputs)} items")
        for i, inp in enumerate(nest_feat_inputs):
            if isinstance(inp, torch.Tensor):
                mean_val = inp.float().mean().item() if inp.numel() > 0 else 0
                print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}, mean={mean_val:.6f}")
    
    # Compare featurizer inputs
    if nemo_feat_inputs and nest_feat_inputs:
        print("\n  Comparing featurizer inputs:")
        for i, (ni, si) in enumerate(zip(nemo_feat_inputs, nest_feat_inputs)):
            if isinstance(ni, torch.Tensor) and isinstance(si, torch.Tensor):
                if ni.shape == si.shape:
                    diff = (ni - si).abs()
                    print(f"    input[{i}]: max_diff={diff.max().item():.6e}")
                else:
                    print(f"    input[{i}]: Shape mismatch! NeMo={ni.shape}, nest={si.shape}")
    
    # Check pre_encode.conv inputs vs preprocessor outputs
    print("\n" + "="*60)
    print("4. Compare preprocessor output vs pre_encode.conv input")
    print("="*60)
    
    nemo_conv_inputs = nemo_layers.get('encoder.pre_encode.conv', {}).get('forward_inputs')
    nest_conv_inputs = nest_layers.get('encoder.pre_encode.conv', {}).get('forward_inputs')
    
    if nemo_prep_outputs and nemo_conv_inputs:
        nemo_prep_out = nemo_prep_outputs[0] if isinstance(nemo_prep_outputs, (list, tuple)) else nemo_prep_outputs
        nemo_conv_in = nemo_conv_inputs[0] if isinstance(nemo_conv_inputs, (list, tuple)) else nemo_conv_inputs
        
        # Simulate transformation
        nemo_simulated = nemo_prep_out.transpose(1, 2).unsqueeze(1)
        
        print(f"NeMo preprocessor output shape: {nemo_prep_out.shape}")
        print(f"NeMo pre_encode.conv input shape: {nemo_conv_in.shape}")
        print(f"NeMo simulated shape: {nemo_simulated.shape}")
        
        if nemo_simulated.shape == nemo_conv_in.shape:
            diff = (nemo_simulated - nemo_conv_in).abs()
            print(f"NeMo simulated vs actual conv input: max_diff={diff.max().item():.6e}")
            
            if diff.max().item() > 1e-5:
                print("  [WARNING] NeMo's conv input doesn't match simulated!")
                print("  This means preprocessor was called multiple times and hook captured a different call.")
    
    if nest_prep_outputs and nest_conv_inputs:
        nest_prep_out = nest_prep_outputs[0] if isinstance(nest_prep_outputs, (list, tuple)) else nest_prep_outputs
        nest_conv_in = nest_conv_inputs[0] if isinstance(nest_conv_inputs, (list, tuple)) else nest_conv_inputs
        
        # Simulate transformation
        nest_simulated = nest_prep_out.transpose(1, 2).unsqueeze(1)
        
        print(f"\nnest preprocessor output shape: {nest_prep_out.shape}")
        print(f"nest pre_encode.conv input shape: {nest_conv_in.shape}")
        print(f"nest simulated shape: {nest_simulated.shape}")
        
        if nest_simulated.shape == nest_conv_in.shape:
            diff = (nest_simulated - nest_conv_in).abs()
            print(f"nest simulated vs actual conv input: max_diff={diff.max().item():.6e}")
            
            if diff.max().item() > 1e-5:
                print("  [WARNING] nest's conv input doesn't match simulated!")
                print("  This means preprocessor was called multiple times and hook captured a different call.")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

