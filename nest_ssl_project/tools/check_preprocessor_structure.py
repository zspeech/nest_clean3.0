#!/usr/bin/env python3
"""
Check preprocessor structure and all its submodules
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
    print("PREPROCESSOR STRUCTURE CHECK")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Find all preprocessor-related keys
    nemo_prep_keys = sorted([k for k in nemo_layers.keys() if 'preprocessor' in k.lower()])
    nest_prep_keys = sorted([k for k in nest_layers.keys() if 'preprocessor' in k.lower()])
    
    print(f"\nNeMo preprocessor keys ({len(nemo_prep_keys)}):")
    for k in nemo_prep_keys:
        print(f"  {k}")
    
    print(f"\nnest preprocessor keys ({len(nest_prep_keys)}):")
    for k in nest_prep_keys:
        print(f"  {k}")
    
    # Check each preprocessor submodule
    print("\n" + "="*60)
    print("Comparing preprocessor submodules")
    print("="*60)
    
    all_keys = sorted(set(nemo_prep_keys) | set(nest_prep_keys))
    
    for key in all_keys:
        print(f"\n--- {key} ---")
        
        nemo_data = nemo_layers.get(key, {})
        nest_data = nest_layers.get(key, {})
        
        nemo_out = nemo_data.get('forward_outputs')
        nest_out = nest_data.get('forward_outputs')
        
        if nemo_out is None and nest_out is None:
            print("  Both None")
            continue
        
        if nemo_out is None:
            print("  NeMo: None")
            continue
        if nest_out is None:
            print("  nest: None")
            continue
        
        # Handle tuple/list outputs
        if isinstance(nemo_out, (list, tuple)):
            nemo_tensor = nemo_out[0]
        else:
            nemo_tensor = nemo_out
            
        if isinstance(nest_out, (list, tuple)):
            nest_tensor = nest_out[0]
        else:
            nest_tensor = nest_out
        
        if not isinstance(nemo_tensor, torch.Tensor) or not isinstance(nest_tensor, torch.Tensor):
            print(f"  NeMo type: {type(nemo_tensor)}, nest type: {type(nest_tensor)}")
            continue
        
        print(f"  NeMo: shape={nemo_tensor.shape}, mean={nemo_tensor.float().mean().item():.6f}")
        print(f"  nest: shape={nest_tensor.shape}, mean={nest_tensor.float().mean().item():.6f}")
        
        if nemo_tensor.shape == nest_tensor.shape:
            diff = (nemo_tensor.float() - nest_tensor.float()).abs()
            max_diff = diff.max().item()
            status = "[OK]" if max_diff < 1e-5 else "[FAIL]"
            print(f"  {status} max_diff={max_diff:.6e}")
        else:
            print(f"  [SHAPE] Shape mismatch!")

    # Check the actual data flow
    print("\n" + "="*60)
    print("Check featurizer output vs preprocessor output")
    print("="*60)
    
    nemo_feat_out = nemo_layers.get('preprocessor.featurizer', {}).get('forward_outputs')
    nest_feat_out = nest_layers.get('preprocessor.featurizer', {}).get('forward_outputs')
    
    nemo_prep_out = nemo_layers.get('preprocessor', {}).get('forward_outputs')
    nest_prep_out = nest_layers.get('preprocessor', {}).get('forward_outputs')
    
    if nemo_feat_out and nemo_prep_out:
        nemo_feat = nemo_feat_out[0] if isinstance(nemo_feat_out, (list, tuple)) else nemo_feat_out
        nemo_prep = nemo_prep_out[0] if isinstance(nemo_prep_out, (list, tuple)) else nemo_prep_out
        
        print(f"\nNeMo featurizer output: shape={nemo_feat.shape}")
        print(f"NeMo preprocessor output: shape={nemo_prep.shape}")
        
        if nemo_feat.shape == nemo_prep.shape:
            diff = (nemo_feat - nemo_prep).abs().max().item()
            print(f"NeMo featurizer vs preprocessor output: max_diff={diff:.6e}")
    
    if nest_feat_out and nest_prep_out:
        nest_feat = nest_feat_out[0] if isinstance(nest_feat_out, (list, tuple)) else nest_feat_out
        nest_prep = nest_prep_out[0] if isinstance(nest_prep_out, (list, tuple)) else nest_prep_out
        
        print(f"\nnest featurizer output: shape={nest_feat.shape}")
        print(f"nest preprocessor output: shape={nest_prep.shape}")
        
        if nest_feat.shape == nest_prep.shape:
            diff = (nest_feat - nest_prep).abs().max().item()
            print(f"nest featurizer vs preprocessor output: max_diff={diff:.6e}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

