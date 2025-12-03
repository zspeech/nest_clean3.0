#!/usr/bin/env python3
"""
Test NeMo's featurizer directly to understand its behavior.
Run this in the NeMo environment to see what NeMo actually does.
"""

import torch
import pickle
from pathlib import Path
import argparse
import sys

# Try to import NeMo
try:
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
    HAVE_NEMO = True
except ImportError:
    HAVE_NEMO = False
    print("NeMo not available. This script must be run in a NeMo environment.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    
    if not HAVE_NEMO:
        print("Cannot run without NeMo. Exiting.")
        return
    
    nemo_dir = Path(args.nemo_dir)
    
    if args.step is None:
        nemo_steps = [int(d.name.split('_')[1]) for d in nemo_dir.glob("step_*") if d.is_dir()]
        if nemo_steps:
            args.step = sorted(nemo_steps)[0]
    
    print("="*80)
    print("TEST NEMO FEATURIZER DIRECTLY")
    print("="*80)
    
    # Load batch
    batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
    audio = batch['audio']
    audio_len = batch['audio_len']
    
    print(f"\n1. Input:")
    print(f"   Audio shape: {audio.shape}")
    print(f"   Audio len: {audio_len.tolist()}")
    
    # Load buffers
    buffers = torch.load(nemo_dir / "buffers" / "buffers.pt", weights_only=False)
    
    # Create featurizer with same config as NeMo
    featurizer = FilterbankFeatures(
        sample_rate=16000,
        n_window_size=400,  # 0.025 * 16000
        n_window_stride=160,  # 0.01 * 16000
        n_fft=512,
        nfilt=80,
        lowfreq=0.0,
        highfreq=8000.0,
        window="hann",
        normalize="per_feature",
        preemph=0.97,
        dither=0.0,  # Disabled for determinism
        pad_to=16,
        mag_power=2.0,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        exact_pad=False,
    )
    
    # Copy buffers from saved
    fb_key = 'preprocessor.featurizer.fb' if 'preprocessor.featurizer.fb' in buffers else 'preprocessor.fb'
    window_key = 'preprocessor.featurizer.window' if 'preprocessor.featurizer.window' in buffers else 'preprocessor.window'
    
    if fb_key in buffers:
        featurizer.fb.data = buffers[fb_key].squeeze(0) if buffers[fb_key].dim() == 3 else buffers[fb_key]
    if window_key in buffers:
        featurizer.window.data = buffers[window_key]
    
    print(f"\n2. Featurizer config:")
    print(f"   n_fft: {featurizer.n_fft}")
    print(f"   hop_length: {featurizer.hop_length}")
    print(f"   win_length: {featurizer.win_length}")
    print(f"   exact_pad: {featurizer.exact_pad}")
    print(f"   stft_pad_amount: {featurizer.stft_pad_amount}")
    print(f"   dither: {featurizer.dither}")
    print(f"   training: {featurizer.training}")
    
    # Set to eval mode
    featurizer.eval()
    print(f"   After eval(): training={featurizer.training}")
    
    # Run featurizer
    print(f"\n3. Running featurizer...")
    with torch.no_grad():
        output, output_len = featurizer(audio, audio_len)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output len: {output_len.tolist()}")
    
    # Load actual saved output
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        layers = pickle.load(f)
    
    actual_output = layers.get('preprocessor.featurizer', {}).get('forward_outputs', [])[0]
    actual_len = layers.get('preprocessor.featurizer', {}).get('forward_outputs', [])[1]
    
    print(f"\n4. Comparison with saved output:")
    print(f"   Saved output shape: {actual_output.shape}")
    print(f"   Saved output len: {actual_len.tolist()}")
    
    # Compare
    diff = (output - actual_output).abs()
    print(f"\n   Direct featurizer vs Saved:")
    print(f"   Max diff: {diff.max().item():.6e}")
    print(f"   Mean diff: {diff.mean().item():.6e}")
    
    print(f"\n   First frame comparison:")
    print(f"   Direct [0, :5, 0]: {output[0, :5, 0].tolist()}")
    print(f"   Saved  [0, :5, 0]: {actual_output[0, :5, 0].tolist()}")
    
    # Now test in training mode
    print(f"\n5. Testing in training mode...")
    featurizer.train()
    print(f"   training={featurizer.training}")
    
    with torch.no_grad():
        output_train, output_len_train = featurizer(audio, audio_len)
    
    diff_train = (output_train - actual_output).abs()
    print(f"\n   Training mode vs Saved:")
    print(f"   Max diff: {diff_train.max().item():.6e}")
    print(f"   Mean diff: {diff_train.mean().item():.6e}")
    
    print(f"\n   First frame (training mode):")
    print(f"   Train  [0, :5, 0]: {output_train[0, :5, 0].tolist()}")
    print(f"   Saved  [0, :5, 0]: {actual_output[0, :5, 0].tolist()}")
    
    print("\n" + "="*80)
    print("Analysis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

