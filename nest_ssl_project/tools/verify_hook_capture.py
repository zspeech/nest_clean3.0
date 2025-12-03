#!/usr/bin/env python3
"""
Verify that hooks are capturing outputs correctly.
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
    print("VERIFY HOOK CAPTURE")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Check preprocessor and featurizer
    layers_to_check = ['preprocessor', 'preprocessor.featurizer', 'encoder', 'decoder']
    
    for layer_name in layers_to_check:
        print(f"\n{layer_name}:")
        
        nemo_data = nemo_layers.get(layer_name, {})
        nest_data = nest_layers.get(layer_name, {})
        
        print(f"  NeMo:")
        print(f"    forward_inputs: {type(nemo_data.get('forward_inputs'))} - len={len(nemo_data.get('forward_inputs', [])) if isinstance(nemo_data.get('forward_inputs'), list) else 'N/A'}")
        print(f"    forward_outputs: {type(nemo_data.get('forward_outputs'))} - len={len(nemo_data.get('forward_outputs', [])) if isinstance(nemo_data.get('forward_outputs'), list) else 'N/A'}")
        
        # Show shapes if available
        fwd_out = nemo_data.get('forward_outputs')
        if fwd_out is not None:
            if isinstance(fwd_out, list) and len(fwd_out) > 0:
                for i, out in enumerate(fwd_out):
                    if isinstance(out, torch.Tensor):
                        print(f"      output[{i}]: shape={out.shape}")
                    else:
                        print(f"      output[{i}]: type={type(out)}")
            elif isinstance(fwd_out, torch.Tensor):
                print(f"      output: shape={fwd_out.shape}")
        
        print(f"  nest:")
        print(f"    forward_inputs: {type(nest_data.get('forward_inputs'))} - len={len(nest_data.get('forward_inputs', [])) if isinstance(nest_data.get('forward_inputs'), list) else 'N/A'}")
        print(f"    forward_outputs: {type(nest_data.get('forward_outputs'))} - len={len(nest_data.get('forward_outputs', [])) if isinstance(nest_data.get('forward_outputs'), list) else 'N/A'}")
        
        fwd_out = nest_data.get('forward_outputs')
        if fwd_out is not None:
            if isinstance(fwd_out, list) and len(fwd_out) > 0:
                for i, out in enumerate(fwd_out):
                    if isinstance(out, torch.Tensor):
                        print(f"      output[{i}]: shape={out.shape}")
                    else:
                        print(f"      output[{i}]: type={type(out)}")
            elif isinstance(fwd_out, torch.Tensor):
                print(f"      output: shape={fwd_out.shape}")
    
    # Check what keys are available in preprocessor.featurizer
    print("\n" + "="*80)
    print("DETAILED CHECK: preprocessor.featurizer")
    print("="*80)
    
    nemo_feat = nemo_layers.get('preprocessor.featurizer', {})
    nest_feat = nest_layers.get('preprocessor.featurizer', {})
    
    print(f"\nNeMo preprocessor.featurizer keys: {list(nemo_feat.keys())}")
    print(f"nest preprocessor.featurizer keys: {list(nest_feat.keys())}")
    
    for key in nemo_feat.keys():
        val = nemo_feat[key]
        if val is None:
            print(f"  NeMo {key}: None")
        elif isinstance(val, list):
            print(f"  NeMo {key}: list of {len(val)} items")
            for i, item in enumerate(val[:3]):  # Show first 3
                if isinstance(item, torch.Tensor):
                    print(f"    [{i}]: Tensor shape={item.shape}")
                else:
                    print(f"    [{i}]: {type(item)}")
        elif isinstance(val, torch.Tensor):
            print(f"  NeMo {key}: Tensor shape={val.shape}")
        else:
            print(f"  NeMo {key}: {type(val)}")
    
    for key in nest_feat.keys():
        val = nest_feat[key]
        if val is None:
            print(f"  nest {key}: None")
        elif isinstance(val, list):
            print(f"  nest {key}: list of {len(val)} items")
            for i, item in enumerate(val[:3]):
                if isinstance(item, torch.Tensor):
                    print(f"    [{i}]: Tensor shape={item.shape}")
                else:
                    print(f"    [{i}]: {type(item)}")
        elif isinstance(val, torch.Tensor):
            print(f"  nest {key}: Tensor shape={val.shape}")
        else:
            print(f"  nest {key}: {type(val)}")
    
    # Now check preprocessor (parent)
    print("\n" + "="*80)
    print("DETAILED CHECK: preprocessor")
    print("="*80)
    
    nemo_preproc = nemo_layers.get('preprocessor', {})
    nest_preproc = nest_layers.get('preprocessor', {})
    
    print(f"\nNeMo preprocessor keys: {list(nemo_preproc.keys())}")
    print(f"nest preprocessor keys: {list(nest_preproc.keys())}")
    
    # Compare preprocessor outputs
    nemo_out = nemo_preproc.get('forward_outputs')
    nest_out = nest_preproc.get('forward_outputs')
    
    if nemo_out is not None and nest_out is not None:
        print("\nPreprocessor outputs comparison:")
        if isinstance(nemo_out, list) and isinstance(nest_out, list):
            print(f"  NeMo: {len(nemo_out)} outputs")
            print(f"  nest: {len(nest_out)} outputs")
            
            for i in range(min(len(nemo_out), len(nest_out))):
                nemo_item = nemo_out[i]
                nest_item = nest_out[i]
                
                if isinstance(nemo_item, torch.Tensor) and isinstance(nest_item, torch.Tensor):
                    if nemo_item.shape == nest_item.shape:
                        diff = (nemo_item - nest_item).abs()
                        print(f"  output[{i}]: shape={nemo_item.shape}, max_diff={diff.max().item():.6e}")
                    else:
                        print(f"  output[{i}]: SHAPE MISMATCH - NeMo={nemo_item.shape}, nest={nest_item.shape}")
                else:
                    print(f"  output[{i}]: NeMo={type(nemo_item)}, nest={type(nest_item)}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

