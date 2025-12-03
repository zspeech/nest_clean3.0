#!/usr/bin/env python3
"""
Diagnose STFT input differences - compare audio after preemphasis but before STFT.
"""

import torch
import pickle
import sys
from pathlib import Path
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Diagnose STFT input")
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
    print("DIAGNOSE STFT INPUT (Audio after preemphasis)")
    print("="*80)
    
    # Load batch data
    try:
        nemo_batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
        nest_batch = torch.load(nest_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
    except Exception as e:
        print(f"Error loading batch: {e}")
        return
    
    # Get audio
    nemo_audio = nemo_batch.get('audio', nemo_batch.get('signal'))
    nest_audio = nest_batch.get('audio', nest_batch.get('signal'))
    nemo_audio_len = nemo_batch.get('audio_len', nemo_batch.get('signal_len'))
    nest_audio_len = nest_batch.get('audio_len', nest_batch.get('signal_len'))
    
    if nemo_audio is None or nest_audio is None:
        print(f"Audio not found in batch")
        print(f"NeMo batch keys: {nemo_batch.keys()}")
        print(f"nest batch keys: {nest_batch.keys()}")
        return
    
    print(f"\n1. Raw Audio:")
    print(f"   NeMo shape: {nemo_audio.shape}, len: {nemo_audio_len.tolist()}")
    print(f"   nest shape: {nest_audio.shape}, len: {nest_audio_len.tolist()}")
    
    diff = (nemo_audio - nest_audio).abs()
    print(f"   Max diff: {diff.max().item():.6e}")
    print(f"   Mean diff: {diff.mean().item():.6e}")
    
    # Apply preemphasis manually
    print(f"\n2. After Preemphasis (manual calculation):")
    preemph = 0.97
    
    def apply_preemph(x, seq_len, preemph_coef):
        """Apply preemphasis exactly as in NeMo/nest."""
        timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len.unsqueeze(1)
        x_preemph = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - preemph_coef * x[:, :-1]), dim=1)
        x_preemph = x_preemph.masked_fill(~timemask, 0.0)
        return x_preemph
    
    nemo_preemph = apply_preemph(nemo_audio.clone(), nemo_audio_len, preemph)
    nest_preemph = apply_preemph(nest_audio.clone(), nest_audio_len, preemph)
    
    diff_preemph = (nemo_preemph - nest_preemph).abs()
    print(f"   Max diff: {diff_preemph.max().item():.6e}")
    print(f"   Mean diff: {diff_preemph.mean().item():.6e}")
    
    # Check first few samples
    print(f"\n3. First 10 Samples of Preemphasized Audio (Sample 0):")
    print(f"   NeMo: {nemo_preemph[0, :10].tolist()}")
    print(f"   nest: {nest_preemph[0, :10].tolist()}")
    
    # Compute STFT manually
    print(f"\n4. Manual STFT Computation:")
    
    # Load buffers to get window
    try:
        nemo_buffers = torch.load(nemo_dir / "buffers" / "buffers.pt", weights_only=False)
        nest_buffers = torch.load(nest_dir / "buffers" / "buffers.pt", weights_only=False)
        
        nemo_window = nemo_buffers.get('preprocessor.featurizer.window') or nemo_buffers.get('preprocessor.window')
        nest_window = nest_buffers.get('preprocessor.featurizer.window') or nest_buffers.get('preprocessor.window')
        
        if nemo_window is not None and nest_window is not None:
            print(f"   Window shapes: NeMo={nemo_window.shape}, nest={nest_window.shape}")
            window_diff = (nemo_window - nest_window).abs().max().item()
            print(f"   Window max diff: {window_diff:.6e}")
    except Exception as e:
        print(f"   Could not load buffers: {e}")
        nemo_window = None
        nest_window = None
    
    # STFT parameters
    n_fft = 512
    hop_length = 160
    win_length = nemo_window.shape[0] if nemo_window is not None else 400
    
    print(f"\n   STFT params: n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}")
    
    # Compute STFT
    window = nemo_window if nemo_window is not None else torch.hann_window(win_length)
    
    nemo_stft = torch.stft(
        nemo_preemph,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        window=window.to(dtype=torch.float, device=nemo_preemph.device),
        return_complex=True,
        pad_mode="constant",
    )
    
    nest_stft = torch.stft(
        nest_preemph,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        window=window.to(dtype=torch.float, device=nest_preemph.device),
        return_complex=True,
        pad_mode="constant",
    )
    
    print(f"\n5. STFT Output:")
    print(f"   NeMo shape: {nemo_stft.shape}")
    print(f"   nest shape: {nest_stft.shape}")
    
    stft_diff = (nemo_stft - nest_stft).abs()
    print(f"   Max diff: {stft_diff.max().item():.6e}")
    print(f"   Mean diff: {stft_diff.mean().item():.6e}")
    
    # Check first frame of STFT
    print(f"\n6. First Frame of STFT (Sample 0):")
    nemo_first_frame = nemo_stft[0, :, 0]
    nest_first_frame = nest_stft[0, :, 0]
    
    first_frame_diff = (nemo_first_frame - nest_first_frame).abs()
    print(f"   Max diff: {first_frame_diff.max().item():.6e}")
    print(f"   Mean diff: {first_frame_diff.mean().item():.6e}")
    
    # Check middle frame
    mid_frame = nemo_stft.shape[2] // 2
    print(f"\n7. Middle Frame of STFT (frame {mid_frame}, Sample 0):")
    nemo_mid_frame = nemo_stft[0, :, mid_frame]
    nest_mid_frame = nest_stft[0, :, mid_frame]
    
    mid_frame_diff = (nemo_mid_frame - nest_mid_frame).abs()
    print(f"   Max diff: {mid_frame_diff.max().item():.6e}")
    print(f"   Mean diff: {mid_frame_diff.mean().item():.6e}")
    
    print("\n" + "="*80)
    print("Diagnosis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

