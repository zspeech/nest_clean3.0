#!/usr/bin/env python3
"""
Diagnose STFT and mel filterbank computation differences.
Compare intermediate outputs between NeMo and nest.
"""

import torch
import pickle
import sys
from pathlib import Path
import argparse


def compare_tensors(name, t1, t2, atol=1e-5, rtol=1e-5):
    """Compare two tensors."""
    if t1 is None or t2 is None:
        print(f"  {name}: One is None")
        return False
    
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        print(f"  {name}: Not tensors: {type(t1)} vs {type(t2)}")
        return False
    
    if t1.shape != t2.shape:
        print(f"  {name}: Shape mismatch: {t1.shape} vs {t2.shape}")
        return False
    
    if t1.dtype != t2.dtype:
        print(f"  {name}: Dtype mismatch: {t1.dtype} vs {t2.dtype}")
    
    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    is_close = torch.allclose(t1, t2, atol=atol, rtol=rtol)
    
    status = "[OK]" if is_close else "[FAIL]"
    print(f"  {name}: {status} Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    
    if not is_close:
        # Find location of max diff
        max_idx = diff.argmax()
        max_idx_unraveled = torch.unravel_index(max_idx, t1.shape)
        print(f"    Max diff at index: {max_idx_unraveled}")
        print(f"    NeMo value: {t1.flatten()[max_idx].item():.6f}")
        print(f"    nest value: {t2.flatten()[max_idx].item():.6f}")
        print(f"    NeMo stats: min={t1.min().item():.6f}, max={t1.max().item():.6f}, mean={t1.mean().item():.6f}")
        print(f"    nest stats: min={t2.min().item():.6f}, max={t2.max().item():.6f}, mean={t2.mean().item():.6f}")
    
    return is_close


def main():
    parser = argparse.ArgumentParser(description="Diagnose STFT differences")
    parser.add_argument("--nemo_dir", type=str, required=True, help="NeMo output directory")
    parser.add_argument("--nest_dir", type=str, required=True, help="nest output directory")
    parser.add_argument("--step", type=int, default=0, help="Step to compare")
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir)
    nest_dir = Path(args.nest_dir)
    
    # Load batch data
    nemo_batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", map_location='cpu', weights_only=False)
    nest_batch = torch.load(nest_dir / f"step_{args.step}" / "batch.pt", map_location='cpu', weights_only=False)
    
    # Load layer outputs
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Load buffers
    nemo_buffers = torch.load(nemo_dir.parent / "buffers" / "buffers.pt", map_location='cpu', weights_only=False)
    nest_buffers = torch.load(nest_dir.parent / "buffers" / "buffers.pt", map_location='cpu', weights_only=False)
    
    print("="*80)
    print("STFT AND MEL FILTERBANK DIAGNOSIS")
    print("="*80)
    
    # Get audio input
    nemo_audio = nemo_batch['audio']
    nest_audio = nest_batch['audio']
    nemo_audio_len = nemo_batch['audio_len']
    nest_audio_len = nest_batch['audio_len']
    
    print("\n1. Audio Input:")
    compare_tensors("audio", nemo_audio, nest_audio)
    compare_tensors("audio_len", nemo_audio_len, nest_audio_len)
    
    # Get featurizer inputs
    if 'preprocessor.featurizer' in nemo_layers and 'preprocessor.featurizer' in nest_layers:
        nemo_feat = nemo_layers['preprocessor.featurizer']
        nest_feat = nest_layers['preprocessor.featurizer']
        
        print("\n2. Featurizer Inputs:")
        nemo_inputs = nemo_feat.get('forward_inputs', [])
        nest_inputs = nest_feat.get('forward_inputs', [])
        
        if nemo_inputs and nest_inputs:
            if len(nemo_inputs) >= 2 and len(nest_inputs) >= 2:
                compare_tensors("input[0] (audio)", nemo_inputs[0], nest_inputs[0])
                compare_tensors("input[1] (seq_len)", nemo_inputs[1], nest_inputs[1])
        
        print("\n3. Featurizer Outputs:")
        nemo_outputs = nemo_feat.get('forward_outputs', [])
        nest_outputs = nest_feat.get('forward_outputs', [])
        
        if nemo_outputs and nest_outputs:
            if len(nemo_outputs) >= 2 and len(nest_outputs) >= 2:
                compare_tensors("output[0] (mel_spec)", nemo_outputs[0], nest_outputs[0])
                compare_tensors("output[1] (seq_len)", nemo_outputs[1], nest_outputs[1])
    
    # Compare buffers
    print("\n4. Buffers:")
    nemo_fb = nemo_buffers.get('preprocessor.featurizer.fb')
    nest_fb = nest_buffers.get('preprocessor.featurizer.fb')
    compare_tensors("fb (mel filterbank)", nemo_fb, nest_fb)
    
    nemo_window = nemo_buffers.get('preprocessor.featurizer.window')
    nest_window = nest_buffers.get('preprocessor.featurizer.window')
    compare_tensors("window", nemo_window, nest_window)
    
    # Manual STFT computation for debugging
    print("\n5. Manual STFT Computation:")
    print("   Computing STFT manually to check intermediate results...")
    
    # Get STFT parameters from buffers (assuming standard values)
    n_fft = 512
    hop_length = 160
    win_length = 320
    
    # Compute STFT manually
    nemo_stft = torch.stft(
        nemo_audio[0],  # First sample
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=True,  # Assuming exact_pad=False
        window=nemo_window.squeeze(0) if nemo_window is not None else None,
        return_complex=True,
        pad_mode="constant",
    )
    
    nest_stft = torch.stft(
        nest_audio[0],  # First sample
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=True,  # Assuming exact_pad=False
        window=nest_window.squeeze(0) if nest_window is not None else None,
        return_complex=True,
        pad_mode="constant",
    )
    
    # Compare complex STFT
    print("   Comparing complex STFT output...")
    compare_tensors("STFT (real)", nemo_stft.real, nest_stft.real, atol=1e-4)
    compare_tensors("STFT (imag)", nemo_stft.imag, nest_stft.imag, atol=1e-4)
    
    # Compare magnitude
    nemo_mag = torch.sqrt(nemo_stft.real.pow(2) + nemo_stft.imag.pow(2))
    nest_mag = torch.sqrt(nest_stft.real.pow(2) + nest_stft.imag.pow(2))
    compare_tensors("Magnitude", nemo_mag, nest_mag, atol=1e-4)
    
    # Compare power spectrum
    nemo_power = nemo_mag.pow(2.0)
    nest_power = nest_mag.pow(2.0)
    compare_tensors("Power spectrum", nemo_power, nest_power, atol=1e-4)
    
    # Compare mel filterbank application
    print("\n6. Mel Filterbank Application:")
    if nemo_fb is not None and nest_fb is not None:
        nemo_mel = torch.matmul(nemo_fb.squeeze(0), nemo_power)
        nest_mel = torch.matmul(nest_fb.squeeze(0), nest_power)
        compare_tensors("Mel spectrogram (before log)", nemo_mel, nest_mel, atol=1e-4)
        
        # Compare log
        log_zero_guard = 2**-24
        nemo_log_mel = torch.log(nemo_mel + log_zero_guard)
        nest_log_mel = torch.log(nest_mel + log_zero_guard)
        compare_tensors("Log mel spectrogram", nemo_log_mel, nest_log_mel, atol=1e-4)
    
    print("\n" + "="*80)
    print("Diagnosis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

