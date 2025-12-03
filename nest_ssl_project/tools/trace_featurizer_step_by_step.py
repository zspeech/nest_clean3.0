#!/usr/bin/env python3
"""
Trace featurizer step by step to find exact divergence point.
Uses saved batch data and buffers to reproduce computation.
"""

import torch
import torch.nn.functional as F
import pickle
from pathlib import Path
import argparse

CONSTANT = 1e-5


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
    print("TRACE FEATURIZER STEP BY STEP")
    print("="*80)
    
    # Load batch
    nemo_batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
    
    # Load buffers
    nemo_buffers = torch.load(nemo_dir / "buffers" / "buffers.pt", weights_only=False)
    
    # Load layer outputs for comparison
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nemo_layers = pickle.load(f)
    with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get actual featurizer outputs
    nemo_feat = nemo_layers.get('preprocessor.featurizer', {})
    nest_feat = nest_layers.get('preprocessor.featurizer', {})
    nemo_actual_out = nemo_feat.get('forward_outputs', [])[0]
    nest_actual_out = nest_feat.get('forward_outputs', [])[0]
    nemo_actual_len = nemo_feat.get('forward_outputs', [])[1]
    
    # Get input
    audio = nemo_batch['audio']
    audio_len = nemo_batch['audio_len']
    
    print(f"\n1. Input:")
    print(f"   Audio shape: {audio.shape}")
    print(f"   Audio len: {audio_len.tolist()}")
    
    # Get buffers
    fb = nemo_buffers.get('preprocessor.featurizer.fb')
    if fb is None:
        fb = nemo_buffers.get('preprocessor.fb')
    window = nemo_buffers.get('preprocessor.featurizer.window')
    if window is None:
        window = nemo_buffers.get('preprocessor.window')
    
    print(f"\n2. Buffers:")
    print(f"   fb shape: {fb.shape if fb is not None else 'None'}")
    print(f"   window shape: {window.shape if window is not None else 'None'}")
    
    # Parameters (matching config)
    n_fft = 512
    hop_length = 160
    win_length = window.shape[0] if window is not None else 400
    preemph = 0.97
    log_zero_guard = 2**-24
    
    # exact_pad=False is the default in NeMo config
    # When exact_pad=False: stft_pad_amount=None, center=True in STFT
    # When exact_pad=True: stft_pad_amount=(n_fft-hop_length)//2, center=False
    exact_pad = False  # Default value, not set in config
    stft_pad_amount = (n_fft - hop_length) // 2 if exact_pad else None
    
    print(f"\n3. Parameters:")
    print(f"   n_fft: {n_fft}")
    print(f"   hop_length: {hop_length}")
    print(f"   win_length: {win_length}")
    print(f"   exact_pad: {exact_pad}")
    print(f"   stft_pad_amount: {stft_pad_amount}")
    
    # Compute seq_len
    # When center=True (stft_pad_amount=None), STFT pads n_fft//2 on each side
    # Output length = 1 + (input_len + 2*pad - n_fft) // hop_length
    #               = 1 + (input_len + n_fft - n_fft) // hop_length
    #               = 1 + input_len // hop_length
    if stft_pad_amount is not None:
        # exact_pad=True, center=False
        pad_amount = stft_pad_amount * 2
        seq_len = torch.floor_divide((audio_len + pad_amount - n_fft), hop_length)
    else:
        # exact_pad=False, center=True
        # STFT with center=True: output_len = 1 + input_len // hop_length
        seq_len = 1 + torch.floor_divide(audio_len, hop_length)
    print(f"   Computed seq_len: {seq_len.tolist()}")
    print(f"   Actual NeMo seq_len: {nemo_actual_len.tolist()}")
    
    # Step-by-step computation
    x = audio.clone()
    seq_len_time = audio_len.clone()
    
    # STFT padding (only when exact_pad=True)
    if stft_pad_amount is not None:
        x = F.pad(x.unsqueeze(1), (stft_pad_amount, stft_pad_amount), "constant").squeeze(1)
        print(f"\n4. After STFT padding: shape={x.shape}")
        center = False
    else:
        print(f"\n4. No manual STFT padding (center=True handles it)")
        center = True
    
    # Preemphasis
    timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
    x_preemph = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - preemph * x[:, :-1]), dim=1)
    x_preemph = x_preemph.masked_fill(~timemask, 0.0)
    print(f"\n5. After preemphasis: shape={x_preemph.shape}")
    print(f"   First 5 values [0]: {x_preemph[0, :5].tolist()}")
    print(f"   Values at boundary [0, seq_len_time-3:seq_len_time+2]: {x_preemph[0, seq_len_time[0]-3:seq_len_time[0]+2].tolist()}")
    
    # STFT
    stft_window = window if window is not None else torch.hann_window(win_length)
    x_stft = torch.stft(
        x_preemph,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=stft_window,
        center=center,  # True when exact_pad=False, False when exact_pad=True
        pad_mode="constant",
        return_complex=True
    )
    print(f"\n6. After STFT: shape={x_stft.shape}")
    
    # Magnitude
    x_real = torch.view_as_real(x_stft)
    x_mag = torch.sqrt(x_real.pow(2).sum(-1))
    print(f"\n7. After magnitude: shape={x_mag.shape}")
    
    # Power spectrum
    x_pow = x_mag.pow(2.0)
    print(f"\n8. After power: shape={x_pow.shape}")
    
    # Mel filterbank
    fb_squeezed = fb.squeeze(0) if fb is not None else None
    if fb_squeezed is not None:
        x_mel = torch.matmul(fb_squeezed, x_pow)
    else:
        x_mel = x_pow
    print(f"\n9. After mel filterbank: shape={x_mel.shape}")
    
    # Log
    x_log = torch.log(x_mel + log_zero_guard)
    print(f"\n10. After log: shape={x_log.shape}")
    
    # Compare with actual output BEFORE normalization
    # We need to compare x_log with actual output before normalize_batch
    
    # Normalize
    batch_size = x_log.shape[0]
    max_time = x_log.shape[2]
    
    time_steps = torch.arange(max_time, device=x_log.device).unsqueeze(0).expand(batch_size, max_time)
    valid_mask = time_steps < seq_len.unsqueeze(1)
    x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x_log, 0.0).sum(axis=2)
    x_mean_denominator = valid_mask.sum(axis=1)
    x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)
    
    x_std = torch.sqrt(
        torch.sum(torch.where(valid_mask.unsqueeze(1), x_log - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
        / (x_mean_denominator.unsqueeze(1) - 1.0)
    )
    x_std = x_std.masked_fill(x_std.isnan(), 0.0)
    x_std += CONSTANT
    
    x_norm = (x_log - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    print(f"\n11. After normalize: shape={x_norm.shape}")
    
    # Mask padding
    mask = torch.arange(max_time, device=x_norm.device)
    mask = mask.repeat(x_norm.size(0), 1) >= seq_len.unsqueeze(1)
    x_final = x_norm.masked_fill(mask.unsqueeze(1).type(torch.bool), 0.0)
    
    # Pad to multiple of 16
    pad_amt = x_final.size(-1) % 16
    if pad_amt != 0:
        x_final = F.pad(x_final, (0, 16 - pad_amt), value=0.0)
    
    print(f"\n12. Final output: shape={x_final.shape}")
    
    # Compare with actual outputs
    print(f"\n" + "="*80)
    print("COMPARISON WITH ACTUAL OUTPUTS")
    print("="*80)
    
    print(f"\n13. Manual vs Actual NeMo:")
    diff_nemo = (x_final - nemo_actual_out).abs()
    print(f"    Max diff: {diff_nemo.max().item():.6e}")
    print(f"    Mean diff: {diff_nemo.mean().item():.6e}")
    
    print(f"\n14. Manual vs Actual nest:")
    diff_nest = (x_final - nest_actual_out).abs()
    print(f"    Max diff: {diff_nest.max().item():.6e}")
    print(f"    Mean diff: {diff_nest.mean().item():.6e}")
    
    print(f"\n15. Actual NeMo vs Actual nest:")
    diff_actual = (nemo_actual_out - nest_actual_out).abs()
    print(f"    Max diff: {diff_actual.max().item():.6e}")
    print(f"    Mean diff: {diff_actual.mean().item():.6e}")
    
    # Check first frame specifically
    print(f"\n16. First frame comparison:")
    print(f"    Manual [0, :5, 0]: {x_final[0, :5, 0].tolist()}")
    print(f"    NeMo   [0, :5, 0]: {nemo_actual_out[0, :5, 0].tolist()}")
    print(f"    nest   [0, :5, 0]: {nest_actual_out[0, :5, 0].tolist()}")
    
    # Check if manual matches nest
    print(f"\n17. Conclusion:")
    if diff_nest.max().item() < 1e-4:
        print("    Manual computation matches nest output!")
        print("    The difference is between NeMo actual and manual/nest.")
        print("    This suggests NeMo has additional processing we're not replicating.")
    elif diff_nemo.max().item() < 1e-4:
        print("    Manual computation matches NeMo output!")
        print("    The difference is in nest's implementation.")
    else:
        print("    Manual computation matches neither!")
        print("    Need to investigate further.")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

