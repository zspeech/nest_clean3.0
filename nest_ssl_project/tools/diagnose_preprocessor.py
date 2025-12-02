#!/usr/bin/env python
"""
Diagnose preprocessor differences between NeMo and nest_ssl_project.
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_dir', type=str, default='./saved_nemo_outputs')
    parser.add_argument('--nest_dir', type=str, default='./saved_nest_outputs')
    parser.add_argument('--step', type=int, default=1)
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir).resolve()
    nest_dir = Path(args.nest_dir).resolve()
    
    # Load layer outputs
    import pickle
    
    nemo_step_dir = nemo_dir / f'step_{args.step}'
    nest_step_dir = nest_dir / f'step_{args.step}'
    
    print(f"Loading layer outputs from step {args.step}...")
    
    with open(nemo_step_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_step_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Find preprocessor layers
    print("\n" + "="*60)
    print("Preprocessor/Featurizer Layer Comparison")
    print("="*60)
    
    for layer_name in sorted(nemo_layers.keys()):
        if 'preprocessor' in layer_name.lower() or 'featurizer' in layer_name.lower():
            print(f"\nNeMo layer: {layer_name}")
            nemo_data = nemo_layers[layer_name]
            nemo_output = nemo_data.get('forward_outputs')
            if isinstance(nemo_output, (list, tuple)):
                for i, out in enumerate(nemo_output):
                    if isinstance(out, torch.Tensor):
                        print(f"  Output[{i}]: shape={list(out.shape)}, dtype={out.dtype}")
                        print(f"    min={out.min().item():.6f}, max={out.max().item():.6f}, mean={out.mean().item():.6f}")
            elif isinstance(nemo_output, torch.Tensor):
                print(f"  Output: shape={list(nemo_output.shape)}, dtype={nemo_output.dtype}")
                print(f"    min={nemo_output.min().item():.6f}, max={nemo_output.max().item():.6f}, mean={nemo_output.mean().item():.6f}")
    
    print("\n" + "-"*60)
    
    for layer_name in sorted(nest_layers.keys()):
        if 'preprocessor' in layer_name.lower() or 'featurizer' in layer_name.lower():
            print(f"\nnest layer: {layer_name}")
            nest_data = nest_layers[layer_name]
            nest_output = nest_data.get('forward_outputs')
            if isinstance(nest_output, (list, tuple)):
                for i, out in enumerate(nest_output):
                    if isinstance(out, torch.Tensor):
                        print(f"  Output[{i}]: shape={list(out.shape)}, dtype={out.dtype}")
                        print(f"    min={out.min().item():.6f}, max={out.max().item():.6f}, mean={out.mean().item():.6f}")
            elif isinstance(nest_output, torch.Tensor):
                print(f"  Output: shape={list(nest_output.shape)}, dtype={nest_output.dtype}")
                print(f"    min={nest_output.min().item():.6f}, max={nest_output.max().item():.6f}, mean={nest_output.mean().item():.6f}")
    
    # Check if both have preprocessor/featurizer
    nemo_preproc = None
    nest_preproc = None
    
    for layer_name in nemo_layers.keys():
        if layer_name == 'preprocessor' or layer_name == 'featurizer':
            nemo_preproc = nemo_layers[layer_name]
            break
    
    for layer_name in nest_layers.keys():
        if layer_name == 'preprocessor' or layer_name == 'featurizer':
            nest_preproc = nest_layers[layer_name]
            break
    
    if nemo_preproc and nest_preproc:
        print("\n" + "="*60)
        print("Direct Comparison")
        print("="*60)
        
        nemo_out = nemo_preproc.get('forward_outputs')
        nest_out = nest_preproc.get('forward_outputs')
        
        if isinstance(nemo_out, (list, tuple)) and isinstance(nest_out, (list, tuple)):
            for i, (n, ne) in enumerate(zip(nemo_out, nest_out)):
                if isinstance(n, torch.Tensor) and isinstance(ne, torch.Tensor):
                    if n.shape == ne.shape:
                        diff = (n - ne).abs()
                        print(f"\nOutput[{i}] comparison:")
                        print(f"  Shape: {list(n.shape)}")
                        print(f"  Max abs diff: {diff.max().item():.6e}")
                        print(f"  Mean abs diff: {diff.mean().item():.6e}")
                        print(f"  Std abs diff: {diff.std().item():.6e}")
                        
                        # Check if values are close
                        is_close = torch.allclose(n, ne, atol=1e-5, rtol=1e-5)
                        print(f"  allclose(atol=1e-5, rtol=1e-5): {is_close}")
                        
                        # Show sample values
                        print(f"  NeMo first 5 values: {n.flatten()[:5].tolist()}")
                        print(f"  nest first 5 values: {ne.flatten()[:5].tolist()}")
                    else:
                        print(f"\nOutput[{i}] shape mismatch: {n.shape} vs {ne.shape}")
    
    # Check librosa version
    print("\n" + "="*60)
    print("Library Versions")
    print("="*60)
    
    try:
        import librosa
        print(f"librosa: {librosa.__version__}")
    except:
        print("librosa: not found")
    
    try:
        import nemo
        print(f"nemo: {nemo.__version__}")
    except:
        print("nemo: not found")
    
    print(f"torch: {torch.__version__}")
    print(f"numpy: {np.__version__}")

if __name__ == '__main__':
    main()

