#!/usr/bin/env python
"""
Detailed diagnosis of featurizer differences between NeMo and nest_ssl_project.
"""
import sys
from pathlib import Path
import torch
import pickle

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))


def extract_tensor_info(tensor):
    """Extract detailed info from a tensor."""
    if not isinstance(tensor, torch.Tensor):
        return str(type(tensor))
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'mean': tensor.float().mean().item(),
        'std': tensor.float().std().item(),
        'first_5': tensor.flatten()[:5].tolist(),
        'last_5': tensor.flatten()[-5:].tolist(),
    }


def compare_outputs(name, nemo_out, nest_out, indent=0):
    """Compare two outputs and print detailed analysis."""
    prefix = "  " * indent
    
    if type(nemo_out) != type(nest_out):
        print(f"{prefix}{name}: Type mismatch - NeMo: {type(nemo_out)}, nest: {type(nest_out)}")
        return False
    
    if isinstance(nemo_out, torch.Tensor) and isinstance(nest_out, torch.Tensor):
        if nemo_out.shape != nest_out.shape:
            print(f"{prefix}{name}: Shape mismatch - NeMo: {nemo_out.shape}, nest: {nest_out.shape}")
            return False
        
        diff = (nemo_out.float() - nest_out.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        if max_diff < 1e-6:
            print(f"{prefix}{name}: [OK] MATCH (max diff: {max_diff:.2e})")
            return True
        else:
            print(f"{prefix}{name}: [FAIL] MISMATCH")
            print(f"{prefix}  Shape: {list(nemo_out.shape)}")
            print(f"{prefix}  Max abs diff: {max_diff:.6e}")
            print(f"{prefix}  Mean abs diff: {mean_diff:.6e}")
            print(f"{prefix}  NeMo - min: {nemo_out.min().item():.6f}, max: {nemo_out.max().item():.6f}, mean: {nemo_out.float().mean().item():.6f}")
            print(f"{prefix}  nest - min: {nest_out.min().item():.6f}, max: {nest_out.max().item():.6f}, mean: {nest_out.float().mean().item():.6f}")
            
            # Find where the max difference is
            max_idx = diff.argmax().item()
            unraveled = []
            temp = max_idx
            for dim in reversed(nemo_out.shape):
                unraveled.insert(0, temp % dim)
                temp //= dim
            print(f"{prefix}  Max diff at index: {unraveled}")
            print(f"{prefix}    NeMo value: {nemo_out.flatten()[max_idx].item():.6f}")
            print(f"{prefix}    nest value: {nest_out.flatten()[max_idx].item():.6f}")
            return False
    
    elif isinstance(nemo_out, (list, tuple)) and isinstance(nest_out, (list, tuple)):
        if len(nemo_out) != len(nest_out):
            print(f"{prefix}{name}: Length mismatch - NeMo: {len(nemo_out)}, nest: {len(nest_out)}")
            return False
        
        all_match = True
        for i, (n, ne) in enumerate(zip(nemo_out, nest_out)):
            if not compare_outputs(f"{name}[{i}]", n, ne, indent):
                all_match = False
        return all_match
    
    elif isinstance(nemo_out, dict) and isinstance(nest_out, dict):
        if set(nemo_out.keys()) != set(nest_out.keys()):
            print(f"{prefix}{name}: Key mismatch")
            print(f"{prefix}  NeMo keys: {sorted(nemo_out.keys())}")
            print(f"{prefix}  nest keys: {sorted(nest_out.keys())}")
            return False
        
        all_match = True
        for key in nemo_out.keys():
            if not compare_outputs(f"{name}['{key}']", nemo_out[key], nest_out[key], indent):
                all_match = False
        return all_match
    
    else:
        if nemo_out == nest_out:
            print(f"{prefix}{name}: [OK] MATCH (non-tensor)")
            return True
        else:
            print(f"{prefix}{name}: [FAIL] MISMATCH (non-tensor)")
            print(f"{prefix}  NeMo: {nemo_out}")
            print(f"{prefix}  nest: {nest_out}")
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_dir', type=str, default='./saved_nemo_outputs')
    parser.add_argument('--nest_dir', type=str, default='./saved_nest_outputs')
    parser.add_argument('--step', type=int, default=None)
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir).resolve()
    nest_dir = Path(args.nest_dir).resolve()
    
    # Auto-detect step
    if args.step is None:
        nemo_steps = [d.name for d in nemo_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
        nest_steps = [d.name for d in nest_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
        common_steps = set(nemo_steps) & set(nest_steps)
        if common_steps:
            step_name = sorted(common_steps, key=lambda x: int(x.split('_')[1]))[0]
            args.step = int(step_name.split('_')[1])
        else:
            print(f"No common steps found. NeMo: {nemo_steps}, nest: {nest_steps}")
            return
    
    nemo_step_dir = nemo_dir / f'step_{args.step}'
    nest_step_dir = nest_dir / f'step_{args.step}'
    
    print(f"Loading layer outputs from step {args.step}...")
    print(f"NeMo dir: {nemo_step_dir}")
    print(f"nest dir: {nest_step_dir}")
    
    with open(nemo_step_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_step_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Find featurizer and preprocessor layers
    print("\n" + "="*80)
    print("FEATURIZER / PREPROCESSOR ANALYSIS")
    print("="*80)
    
    featurizer_names = ['featurizer', 'preprocessor', 'preprocessor.featurizer']
    
    for layer_name in featurizer_names:
        print(f"\n{'='*60}")
        print(f"Layer: {layer_name}")
        print('='*60)
        
        nemo_data = nemo_layers.get(layer_name)
        nest_data = nest_layers.get(layer_name)
        
        if nemo_data is None:
            print(f"  NeMo: NOT FOUND")
        else:
            print(f"  NeMo: FOUND")
            
        if nest_data is None:
            print(f"  nest: NOT FOUND")
        else:
            print(f"  nest: FOUND")
        
        if nemo_data is None or nest_data is None:
            continue
        
        # Compare forward inputs
        print(f"\n  Forward Inputs:")
        nemo_inputs = nemo_data.get('forward_inputs')
        nest_inputs = nest_data.get('forward_inputs')
        compare_outputs("inputs", nemo_inputs, nest_inputs, indent=2)
        
        # Compare forward outputs
        print(f"\n  Forward Outputs:")
        nemo_outputs = nemo_data.get('forward_outputs')
        nest_outputs = nest_data.get('forward_outputs')
        compare_outputs("outputs", nemo_outputs, nest_outputs, indent=2)
    
    # Also check for any layer containing 'featurizer' or 'preprocessor'
    print("\n" + "="*80)
    print("ALL LAYERS CONTAINING 'featurizer' OR 'preprocessor'")
    print("="*80)
    
    all_names = set(nemo_layers.keys()) | set(nest_layers.keys())
    relevant_names = sorted([n for n in all_names if 'featurizer' in n.lower() or 'preprocessor' in n.lower()])
    
    for layer_name in relevant_names:
        if layer_name in featurizer_names:
            continue
        
        print(f"\n{layer_name}:")
        nemo_data = nemo_layers.get(layer_name)
        nest_data = nest_layers.get(layer_name)
        
        if nemo_data is None:
            print(f"  NeMo: NOT FOUND")
            continue
        if nest_data is None:
            print(f"  nest: NOT FOUND")
            continue
        
        nemo_outputs = nemo_data.get('forward_outputs')
        nest_outputs = nest_data.get('forward_outputs')
        compare_outputs("outputs", nemo_outputs, nest_outputs, indent=1)
    
    # Check batch input data
    print("\n" + "="*80)
    print("BATCH INPUT DATA")
    print("="*80)
    
    nemo_batch = torch.load(nemo_step_dir / 'batch.pt', map_location='cpu', weights_only=False)
    nest_batch = torch.load(nest_step_dir / 'batch.pt', map_location='cpu', weights_only=False)
    
    # Extract audio data
    if isinstance(nemo_batch, dict):
        nemo_audio = nemo_batch.get('noisy_audio') if nemo_batch.get('noisy_audio') is not None else nemo_batch.get('audio')
        nemo_len = nemo_batch.get('noisy_audio_len') if nemo_batch.get('noisy_audio_len') is not None else nemo_batch.get('audio_len')
    else:
        nemo_audio = getattr(nemo_batch, 'noisy_audio', None)
        if nemo_audio is None:
            nemo_audio = getattr(nemo_batch, 'audio', None)
        nemo_len = getattr(nemo_batch, 'noisy_audio_len', None)
        if nemo_len is None:
            nemo_len = getattr(nemo_batch, 'audio_len', None)
    
    if isinstance(nest_batch, dict):
        nest_audio = nest_batch.get('noisy_audio') if nest_batch.get('noisy_audio') is not None else nest_batch.get('audio')
        nest_len = nest_batch.get('noisy_audio_len') if nest_batch.get('noisy_audio_len') is not None else nest_batch.get('audio_len')
    else:
        nest_audio = getattr(nest_batch, 'noisy_audio', None)
        if nest_audio is None:
            nest_audio = getattr(nest_batch, 'audio', None)
        nest_len = getattr(nest_batch, 'noisy_audio_len', None)
        if nest_len is None:
            nest_len = getattr(nest_batch, 'audio_len', None)
    
    print("\nAudio input:")
    compare_outputs("audio", nemo_audio, nest_audio, indent=0)
    
    print("\nAudio length:")
    compare_outputs("audio_len", nemo_len, nest_len, indent=0)
    
    # Check featurizer internals (fb, window)
    print("\n" + "="*80)
    print("FEATURIZER INTERNALS")
    print("="*80)
    
    if 'preprocessor.featurizer' in nemo_layers and 'preprocessor.featurizer' in nest_layers:
        print("\nMel Filterbank (fb):")
        nemo_fb = nemo_layers['preprocessor.featurizer'].get('fb')
        nest_fb = nest_layers['preprocessor.featurizer'].get('fb')
        if nemo_fb is not None and nest_fb is not None:
            compare_outputs("fb", nemo_fb, nest_fb, indent=1)
        else:
            print("  Not found in layer outputs")
            
        print("\nWindow Function (window):")
        nemo_window = nemo_layers['preprocessor.featurizer'].get('window')
        nest_window = nest_layers['preprocessor.featurizer'].get('window')
        if nemo_window is not None and nest_window is not None:
            compare_outputs("window", nemo_window, nest_window, indent=1)
        else:
            print("  Not found in layer outputs")
    
    # Check encoder layers
    print("\n" + "="*80)
    print("ENCODER LAYER ANALYSIS")
    print("="*80)
    
    encoder_layers = sorted([n for n in nemo_layers.keys() if n.startswith('encoder') or n.startswith('pre_encode')])
    
    for layer_name in encoder_layers[:20]:  # First 20 encoder layers
        nemo_data = nemo_layers.get(layer_name)
        nest_data = nest_layers.get(layer_name)
        
        if nemo_data is None or nest_data is None:
            continue
        
        nemo_out = nemo_data.get('forward_outputs')
        nest_out = nest_data.get('forward_outputs')
        
        print(f"\n{layer_name}:")
        compare_outputs("output", nemo_out, nest_out, indent=1)


if __name__ == '__main__':
    main()

