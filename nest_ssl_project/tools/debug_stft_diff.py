#!/usr/bin/env python3
"""
Debug STFT differences between NeMo and nest.
"""

import torch
import pickle
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir)
    
    if args.step is None:
        nemo_steps = [int(d.name.split('_')[1]) for d in nemo_dir.glob("step_*") if d.is_dir()]
        if nemo_steps:
            args.step = sorted(nemo_steps)[0]
    
    print("="*80)
    print("DEBUG STFT DIFFERENCES")
    print("="*80)
    
    # Load batch
    batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
    audio = batch['audio']
    audio_len = batch['audio_len']
    
    # Load buffers
    buffers = torch.load(nemo_dir / "buffers" / "buffers.pt", weights_only=False)
    
    window_key = 'preprocessor.featurizer.window' if 'preprocessor.featurizer.window' in buffers else 'preprocessor.window'
    window = buffers[window_key]
    
    print(f"\n1. Input:")
    print(f"   Audio shape: {audio.shape}")
    print(f"   Audio len: {audio_len.tolist()}")
    print(f"   Window shape: {window.shape}")
    
    # Parameters
    n_fft = 512
    hop_length = 160
    win_length = window.shape[0]
    preemph = 0.97
    
    print(f"\n2. STFT parameters:")
    print(f"   n_fft: {n_fft}")
    print(f"   hop_length: {hop_length}")
    print(f"   win_length: {win_length}")
    print(f"   center: True")
    print(f"   pad_mode: constant")
    
    # Preemphasis
    x = audio.clone()
    seq_len_time = audio_len.clone()
    timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
    x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - preemph * x[:, :-1]), dim=1)
    x = x.masked_fill(~timemask, 0.0)
    
    print(f"\n3. After preemphasis:")
    print(f"   Shape: {x.shape}")
    
    # Test STFT with different window configurations
    print(f"\n4. Testing STFT variants:")
    
    # Variant 1: window as float32
    window_f32 = window.to(dtype=torch.float32)
    stft1 = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_f32,
        center=True,
        pad_mode="constant",
        return_complex=True
    )
    print(f"   Variant 1 (float32 window): shape={stft1.shape}")
    
    # Variant 2: window moved to same device as x
    window_device = window.to(dtype=torch.float, device=x.device)
    stft2 = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_device,
        center=True,
        pad_mode="constant",
        return_complex=True
    )
    print(f"   Variant 2 (device matched): shape={stft2.shape}")
    
    # Compare variants
    diff12 = (stft1 - stft2).abs()
    print(f"\n   Variant 1 vs 2: max_diff={diff12.max().item():.6e}")
    
    # Check boundary frames specifically
    print(f"\n5. Boundary frame analysis:")
    
    # First frame
    print(f"   First frame (t=0):")
    print(f"     stft1 real[0, :5, 0]: {stft1[0, :5, 0].real.tolist()}")
    print(f"     stft1 imag[0, :5, 0]: {stft1[0, :5, 0].imag.tolist()}")
    
    # Last frame for sample 0
    last_frame_0 = audio_len[0].item() // hop_length
    print(f"\n   Last frame for sample 0 (t={last_frame_0}):")
    if last_frame_0 < stft1.shape[2]:
        print(f"     stft1 real[0, :5, {last_frame_0}]: {stft1[0, :5, last_frame_0].real.tolist()}")
    
    # Last frame for sample 1
    last_frame_1 = audio_len[1].item() // hop_length
    print(f"\n   Last frame for sample 1 (t={last_frame_1}):")
    if last_frame_1 < stft1.shape[2]:
        print(f"     stft1 real[1, :5, {last_frame_1}]: {stft1[1, :5, last_frame_1].real.tolist()}")
    
    # Check actual last frame in STFT output
    actual_last = stft1.shape[2] - 1
    print(f"\n   Actual last STFT frame (t={actual_last}):")
    print(f"     stft1 real[0, :5, {actual_last}]: {stft1[0, :5, actual_last].real.tolist()}")
    print(f"     stft1 real[1, :5, {actual_last}]: {stft1[1, :5, actual_last].real.tolist()}")
    
    # Check if the input to STFT has zeros at boundaries
    print(f"\n6. Input to STFT at boundaries:")
    print(f"   Sample 0 first 5: {x[0, :5].tolist()}")
    print(f"   Sample 0 around seq_len ({audio_len[0].item()}):")
    print(f"     x[0, {audio_len[0].item()-3}:{audio_len[0].item()+2}]: {x[0, audio_len[0].item()-3:audio_len[0].item()+2].tolist()}")
    print(f"   Sample 1 around seq_len ({audio_len[1].item()}):")
    print(f"     x[1, {audio_len[1].item()-3}:{audio_len[1].item()+2}]: {x[1, audio_len[1].item()-3:min(audio_len[1].item()+2, x.shape[1])].tolist()}")
    
    # Test with different input lengths
    print(f"\n7. Test STFT output length formula:")
    for sample_idx in range(2):
        input_len = x.shape[1]
        expected_len = 1 + input_len // hop_length
        actual_len = stft1.shape[2]
        print(f"   Sample {sample_idx}: input_len={input_len}, expected_output={expected_len}, actual_output={actual_len}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

