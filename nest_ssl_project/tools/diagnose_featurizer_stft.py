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
    
    # Convert to float for comparison to handle integer types (e.g. Long)
    t1_float = t1.float()
    t2_float = t2.float()
    
    diff = (t1_float - t2_float).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    is_close = torch.allclose(t1_float, t2_float, atol=atol, rtol=rtol)
    
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
    parser.add_argument("--step", type=int, default=None, help="Step to compare (default: auto-detect)")
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir)
    nest_dir = Path(args.nest_dir)
    
    # Auto-detect step if not provided
    if args.step is None:
        # Try to find common steps
        nemo_steps = [int(d.name.split('_')[1]) for d in nemo_dir.glob("step_*") if d.is_dir()]
        nest_steps = [int(d.name.split('_')[1]) for d in nest_dir.glob("step_*") if d.is_dir()]
        
        common_steps = sorted(list(set(nemo_steps) & set(nest_steps)))
        
        if not common_steps:
            print(f"No common steps found in {nemo_dir} and {nest_dir}")
            # If no common steps, try to use any available step
            if nemo_steps:
                args.step = sorted(nemo_steps)[0]
                print(f"Using NeMo step: {args.step}")
            else:
                print("No steps found.")
                return
        else:
            args.step = common_steps[0]
            print(f"Auto-detected common step: {args.step}")
    
    print(f"Analyzing step: {args.step}")
    
    # Load batch data
    try:
        nemo_batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", map_location='cpu', weights_only=False)
        nest_batch = torch.load(nest_dir / f"step_{args.step}" / "batch.pt", map_location='cpu', weights_only=False)
    except FileNotFoundError as e:
        print(f"Error loading batch data: {e}")
        print(f"Please check if step_{args.step} exists in both directories.")
        return
    
    # Load layer outputs
    try:
        with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
            nemo_layers = pickle.load(f)
        with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
            nest_layers = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading layer outputs: {e}")
        return
    
    # Load buffers
    try:
        nemo_buffers = torch.load(nemo_dir / "buffers" / "buffers.pt", map_location='cpu', weights_only=False)
        nest_buffers = torch.load(nest_dir / "buffers" / "buffers.pt", map_location='cpu', weights_only=False)
    except FileNotFoundError as e:
        print(f"Error loading buffers: {e}")
        print(f"Please check if buffers/buffers.pt exists in output directories.")
        # Continue without buffers if possible, but manual STFT will fail
        nemo_buffers = {}
        nest_buffers = {}
    
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
    
    # Get STFT parameters from buffers
    n_fft = 512
    hop_length = 160
    
    # Detect win_length from window buffer
    if nemo_window is not None:
        win_length = nemo_window.numel()
        print(f"   Detected win_length from buffer: {win_length}")
    else:
        win_length = 320
        print(f"   Using default win_length: {win_length}")
    
    # Compute STFT manually
    try:
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
    except Exception as e:
        print(f"   NeMo STFT computation failed: {e}")
        nemo_stft = None

    try:
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
    except Exception as e:
        print(f"   nest STFT computation failed: {e}")
        nest_stft = None
    
    if nemo_stft is not None and nest_stft is not None:
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
            # fb is usually [1, n_mels, n_fft//2 + 1], power is [n_fft//2 + 1, T]
            # Need to squeeze fb to [n_mels, n_fft//2 + 1]
            fb_tensor = nemo_fb.squeeze(0) if nemo_fb.dim() == 3 else nemo_fb
            
            nemo_mel = torch.matmul(fb_tensor, nemo_power)
            nest_mel = torch.matmul(fb_tensor, nest_power) # Use same fb for fair comparison
            compare_tensors("Mel spectrogram (before log)", nemo_mel, nest_mel, atol=1e-4)
            
            # Compare log
            log_zero_guard = 2**-24
            nemo_log_mel = torch.log(nemo_mel + log_zero_guard)
            nest_log_mel = torch.log(nest_mel + log_zero_guard)
            compare_tensors("Log mel spectrogram", nemo_log_mel, nest_log_mel, atol=1e-4)
            
            # Compare Normalization
            print("\n7. Normalization (per_feature):")
            # Simulate normalization on the single sample we computed
            # Input: [D, T]
            x = nemo_log_mel
            seq_len = x.shape[1]
            
            # Calculate mean and std manually
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            
            # Note: NeMo's normalize_batch uses a slightly different std calculation (biased estimator + subtract 1 in denom?)
            # Let's try to match NeMo's logic exactly
            # x_std = sqrt( sum((x - mean)**2) / (N - 1) )  <-- standard unbiased
            # NeMo: sqrt( sum(...) / (N - 1) ) + 1e-5
            
            norm_x = (x - mean) / (std + 1e-5)
            
            print(f"   Simulated Normalized Output stats:")
            print(f"     mean: {norm_x.mean().item():.6f}")
            print(f"     std: {norm_x.std().item():.6f}")
            print(f"     min: {norm_x.min().item():.6f}")
            print(f"     max: {norm_x.max().item():.6f}")
            
            # Compare with actual featurizer output (first sample, unpadded region)
            if nemo_outputs and len(nemo_outputs) >= 2:
                actual_out = nemo_outputs[0][0] # First sample
                actual_len = nemo_outputs[1][0].item()
                
                # Slice actual output to valid length
                actual_out_valid = actual_out[:, :actual_len]
                
                # Compare shapes
                print(f"   Comparing simulated vs actual:")
                print(f"     Simulated shape: {norm_x.shape}")
                print(f"     Actual valid shape: {actual_out_valid.shape}")
                
                if norm_x.shape == actual_out_valid.shape:
                    compare_tensors("Simulated vs Actual NeMo Output", norm_x, actual_out_valid, atol=1e-3)
                else:
                    print("     [SKIP] Shape mismatch, cannot compare directly")

    else:
        print("   Skipping STFT comparison due to computation failure.")
    else:
        print("   Skipping STFT comparison due to computation failure.")
    
    print("\n" + "="*80)
    print("Diagnosis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

