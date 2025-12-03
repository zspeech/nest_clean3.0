#!/usr/bin/env python3
"""
Diagnose layer count difference between NeMo and nest.
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
    print("LAYER COUNT DIFFERENCE DIAGNOSIS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    nemo_keys = set(nemo_layers.keys())
    nest_keys = set(nest_layers.keys())
    
    print(f"\n1. Layer counts:")
    print(f"   NeMo: {len(nemo_keys)}")
    print(f"   nest: {len(nest_keys)}")
    
    # Find layers only in NeMo
    nemo_only = nemo_keys - nest_keys
    print(f"\n2. Layers ONLY in NeMo ({len(nemo_only)}):")
    for layer in sorted(nemo_only):
        print(f"   - {layer}")
    
    # Find layers only in nest
    nest_only = nest_keys - nemo_keys
    print(f"\n3. Layers ONLY in nest ({len(nest_only)}):")
    for layer in sorted(nest_only):
        print(f"   - {layer}")
    
    # Check preprocessor-related layers
    print(f"\n4. Preprocessor-related layers:")
    print("   NeMo:")
    for layer in sorted(nemo_keys):
        if 'preprocessor' in layer.lower() or 'featurizer' in layer.lower():
            print(f"     - {layer}")
    print("   nest:")
    for layer in sorted(nest_keys):
        if 'preprocessor' in layer.lower() or 'featurizer' in layer.lower():
            print(f"     - {layer}")
    
    # Check if the extra NeMo layers are related to preprocessor
    print(f"\n5. Analysis of NeMo-only layers:")
    preproc_only = [l for l in nemo_only if 'preprocessor' in l.lower() or 'featurizer' in l.lower()]
    encoder_only = [l for l in nemo_only if 'encoder' in l.lower()]
    decoder_only = [l for l in nemo_only if 'decoder' in l.lower()]
    other_only = [l for l in nemo_only if l not in preproc_only and l not in encoder_only and l not in decoder_only]
    
    print(f"   Preprocessor-related: {len(preproc_only)}")
    for l in sorted(preproc_only):
        print(f"     - {l}")
    print(f"   Encoder-related: {len(encoder_only)}")
    for l in sorted(encoder_only):
        print(f"     - {l}")
    print(f"   Decoder-related: {len(decoder_only)}")
    for l in sorted(decoder_only):
        print(f"     - {l}")
    print(f"   Other: {len(other_only)}")
    for l in sorted(other_only):
        print(f"     - {l}")
    
    # Compare featurizer outputs in detail
    print(f"\n6. Featurizer output comparison:")
    nemo_feat = nemo_layers.get('preprocessor.featurizer', {})
    nest_feat = nest_layers.get('preprocessor.featurizer', {})
    
    nemo_out = nemo_feat.get('forward_output', [])
    nest_out = nest_feat.get('forward_output', [])
    
    print(f"   NeMo forward_output length: {len(nemo_out)}")
    print(f"   nest forward_output length: {len(nest_out)}")
    
    if len(nemo_out) >= 2 and len(nest_out) >= 2:
        nemo_spec = nemo_out[0]
        nest_spec = nest_out[0]
        nemo_len = nemo_out[1]
        nest_len = nest_out[1]
        
        print(f"\n   Spectrogram:")
        print(f"     NeMo shape: {nemo_spec.shape}")
        print(f"     nest shape: {nest_spec.shape}")
        
        print(f"\n   Sequence length:")
        print(f"     NeMo: {nemo_len}")
        print(f"     nest: {nest_len}")
        
        if nemo_spec.shape == nest_spec.shape:
            diff = (nemo_spec - nest_spec).abs()
            print(f"\n   Difference analysis:")
            print(f"     Max diff: {diff.max().item():.6e}")
            print(f"     Mean diff: {diff.mean().item():.6e}")
            
            # Check per-sample
            for b in range(nemo_spec.shape[0]):
                valid_len = min(nemo_len[b].item(), nest_len[b].item())
                sample_diff = (nemo_spec[b, :, :valid_len] - nest_spec[b, :, :valid_len]).abs()
                print(f"\n     Sample {b} (valid_len={valid_len}):")
                print(f"       Max diff: {sample_diff.max().item():.6e}")
                print(f"       Mean diff: {sample_diff.mean().item():.6e}")
                
                # Check where max diff is
                max_idx = sample_diff.argmax()
                feat_idx = max_idx // valid_len
                time_idx = max_idx % valid_len
                print(f"       Max diff at: feat={feat_idx}, time={time_idx}")
                print(f"       NeMo value: {nemo_spec[b, feat_idx, time_idx].item():.6f}")
                print(f"       nest value: {nest_spec[b, feat_idx, time_idx].item():.6f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

