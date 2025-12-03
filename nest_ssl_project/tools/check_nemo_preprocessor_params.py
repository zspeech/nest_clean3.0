#!/usr/bin/env python3
"""
Check NeMo preprocessor parameters from saved layer outputs.
"""

import torch
import pickle
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Check NeMo preprocessor params")
    parser.add_argument("--nemo_dir", type=str, required=True, help="NeMo output directory")
    parser.add_argument("--nest_dir", type=str, required=True, help="nest output directory")
    parser.add_argument("--step", type=int, default=None, help="Step to compare")
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir)
    nest_dir = Path(args.nest_dir)
    
    # Auto-detect step
    if args.step is None:
        nemo_steps = [int(d.name.split('_')[1]) for d in nemo_dir.glob("step_*") if d.is_dir()]
        nest_steps = [int(d.name.split('_')[1]) for d in nest_dir.glob("step_*") if d.is_dir()]
        common_steps = sorted(list(set(nemo_steps) & set(nest_steps)))
        if common_steps:
            args.step = common_steps[0]
            print(f"Auto-detected step: {args.step}")
        else:
            print("No common steps found.")
            return
    
    print("="*80)
    print("CHECK PREPROCESSOR PARAMETERS")
    print("="*80)
    
    # Check if model_structure.txt exists
    nemo_structure = nemo_dir / "model_structure.txt"
    nest_structure = nest_dir / "model_structure.txt"
    
    if nemo_structure.exists():
        print(f"\n1. NeMo Model Structure (first 100 lines):")
        with open(nemo_structure, 'r') as f:
            lines = f.readlines()[:100]
            for line in lines:
                if 'preprocessor' in line.lower() or 'featurizer' in line.lower():
                    print(f"   {line.rstrip()}")
    
    if nest_structure.exists():
        print(f"\n2. nest Model Structure (first 100 lines):")
        with open(nest_structure, 'r') as f:
            lines = f.readlines()[:100]
            for line in lines:
                if 'preprocessor' in line.lower() or 'featurizer' in line.lower():
                    print(f"   {line.rstrip()}")
    
    # Load layer outputs to check if there are any clues
    print(f"\n3. Loading layer outputs...")
    try:
        with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
            nemo_layers = pickle.load(f)
        with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
            nest_layers = pickle.load(f)
        
        # Find preprocessor-related layers
        nemo_preproc_layers = [k for k in nemo_layers.keys() if 'preprocessor' in k.lower() or 'featurizer' in k.lower()]
        nest_preproc_layers = [k for k in nest_layers.keys() if 'preprocessor' in k.lower() or 'featurizer' in k.lower()]
        
        print(f"   NeMo preprocessor layers: {nemo_preproc_layers}")
        print(f"   nest preprocessor layers: {nest_preproc_layers}")
        
    except Exception as e:
        print(f"   Error loading layer outputs: {e}")
    
    # Load buffers and check for differences
    print(f"\n4. Checking buffers...")
    try:
        nemo_buffers = torch.load(nemo_dir / "buffers" / "buffers.pt", weights_only=False)
        nest_buffers = torch.load(nest_dir / "buffers" / "buffers.pt", weights_only=False)
        
        # Compare all buffers
        all_keys = set(nemo_buffers.keys()) | set(nest_buffers.keys())
        preproc_keys = [k for k in all_keys if 'preprocessor' in k.lower() or 'featurizer' in k.lower()]
        
        print(f"   Preprocessor buffer keys: {preproc_keys}")
        
        for key in preproc_keys:
            nemo_val = nemo_buffers.get(key)
            nest_val = nest_buffers.get(key)
            
            if isinstance(nemo_val, torch.Tensor) and isinstance(nest_val, torch.Tensor):
                diff = (nemo_val - nest_val).abs().max().item()
                print(f"   {key}: shape={nemo_val.shape}, max_diff={diff:.6e}")
            else:
                print(f"   {key}: NeMo={type(nemo_val)}, nest={type(nest_val)}")
                
    except Exception as e:
        print(f"   Error loading buffers: {e}")
    
    # Check if training mode affects the output
    print(f"\n5. Checking training mode effects...")
    print("   Note: dither and nb_augmentation only apply in training mode")
    print("   If NeMo was in training mode but nest was in eval mode, outputs would differ")
    
    # Load batch to check if inputs are identical
    print(f"\n6. Verifying batch inputs are identical...")
    try:
        nemo_batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
        nest_batch = torch.load(nest_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
        
        nemo_audio = nemo_batch.get('audio')
        nest_audio = nest_batch.get('audio')
        
        if isinstance(nemo_audio, torch.Tensor) and isinstance(nest_audio, torch.Tensor):
            audio_diff = (nemo_audio - nest_audio).abs().max().item()
            print(f"   Audio max diff: {audio_diff:.6e}")
        
    except Exception as e:
        print(f"   Error loading batch: {e}")
    
    print("\n" + "="*80)
    print("Check complete.")
    print("="*80)


if __name__ == '__main__':
    main()

