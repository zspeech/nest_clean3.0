#!/usr/bin/env python3
"""
Diagnose normalize_batch internal calculations.
Compare intermediate values between NeMo and nest.
"""

import torch
import pickle
import sys
from pathlib import Path
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

CONSTANT = 1e-5


def normalize_batch_debug(x, seq_len, normalize_type="per_feature"):
    """normalize_batch with debug output."""
    debug = {}
    
    if normalize_type == "per_feature":
        batch_size = x.shape[0]
        max_time = x.shape[2]
        
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        valid_mask = time_steps < seq_len.unsqueeze(1)
        
        debug['time_steps'] = time_steps.clone()
        debug['valid_mask'] = valid_mask.clone()
        debug['valid_mask_sum'] = valid_mask.sum(dim=1).clone()
        
        x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
        x_mean_denominator = valid_mask.sum(axis=1)
        x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)
        
        debug['x_mean_numerator'] = x_mean_numerator.clone()
        debug['x_mean_denominator'] = x_mean_denominator.clone()
        debug['x_mean'] = x_mean.clone()
        
        # Subtract 1 in the denominator to correct for the bias.
        diff_sq = torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2
        x_std = torch.sqrt(
            torch.sum(diff_sq, axis=2)
            / (x_mean_denominator.unsqueeze(1) - 1.0)
        )
        x_std = x_std.masked_fill(x_std.isnan(), 0.0)
        x_std += CONSTANT
        
        debug['diff_sq_sum'] = torch.sum(diff_sq, axis=2).clone()
        debug['x_std_before_constant'] = (x_std - CONSTANT).clone()
        debug['x_std'] = x_std.clone()
        
        result = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
        debug['result'] = result.clone()
        
        return result, x_mean, x_std, debug
    
    return x, None, None, {}


def main():
    parser = argparse.ArgumentParser(description="Diagnose normalize_batch internal")
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
    print("DIAGNOSE NORMALIZE_BATCH INTERNAL")
    print("="*80)
    
    # Load batch data
    try:
        nemo_batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
        nest_batch = torch.load(nest_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
    except Exception as e:
        print(f"Error loading batch: {e}")
        return
    
    # Get audio
    nemo_audio = nemo_batch.get('audio')
    nest_audio = nest_batch.get('audio')
    nemo_audio_len = nemo_batch.get('audio_len')
    nest_audio_len = nest_batch.get('audio_len')
    
    if nemo_audio is None:
        print(f"Audio not found. Keys: {nemo_batch.keys()}")
        return
    
    print(f"\n1. Input Audio:")
    print(f"   NeMo shape: {nemo_audio.shape}, len: {nemo_audio_len.tolist()}")
    print(f"   nest shape: {nest_audio.shape}, len: {nest_audio_len.tolist()}")
    
    # Load buffers
    try:
        nemo_buffers = torch.load(nemo_dir / "buffers" / "buffers.pt", weights_only=False)
        nest_buffers = torch.load(nest_dir / "buffers" / "buffers.pt", weights_only=False)
        
        # Get fb buffer
        nemo_fb = nemo_buffers.get('preprocessor.featurizer.fb')
        if nemo_fb is None:
            nemo_fb = nemo_buffers.get('preprocessor.fb')
        
        nest_fb = nest_buffers.get('preprocessor.featurizer.fb')
        if nest_fb is None:
            nest_fb = nest_buffers.get('preprocessor.fb')
        
        # Get window buffer
        nemo_window = nemo_buffers.get('preprocessor.featurizer.window')
        if nemo_window is None:
            nemo_window = nemo_buffers.get('preprocessor.window')
        
        nest_window = nest_buffers.get('preprocessor.featurizer.window')
        if nest_window is None:
            nest_window = nest_buffers.get('preprocessor.window')
        
        print(f"\n2. Buffers loaded successfully")
        print(f"   Available buffer keys: {list(nemo_buffers.keys())[:10]}...")
        print(f"   fb shape: {nemo_fb.shape if isinstance(nemo_fb, torch.Tensor) else 'None'}")
        print(f"   window shape: {nemo_window.shape if isinstance(nemo_window, torch.Tensor) else 'None'}")
    except Exception as e:
        print(f"\n2. Error loading buffers: {e}")
        return
    
    # Compute Log Mel manually
    print(f"\n3. Computing Log Mel manually...")
    
    # Parameters
    preemph = 0.97
    n_fft = 512
    hop_length = 160
    win_length = nemo_window.shape[0] if isinstance(nemo_window, torch.Tensor) else 400
    log_zero_guard = 2**-24
    
    # Apply preemphasis
    def apply_preemph(x, seq_len, preemph_coef):
        timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len.unsqueeze(1)
        x_preemph = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - preemph_coef * x[:, :-1]), dim=1)
        x_preemph = x_preemph.masked_fill(~timemask, 0.0)
        return x_preemph
    
    nemo_preemph = apply_preemph(nemo_audio.clone(), nemo_audio_len, preemph)
    nest_preemph = apply_preemph(nest_audio.clone(), nest_audio_len, preemph)
    
    # STFT
    window = nemo_window if isinstance(nemo_window, torch.Tensor) else torch.hann_window(win_length)
    
    nemo_stft = torch.stft(
        nemo_preemph, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        center=True, window=window.float(), return_complex=True, pad_mode="constant"
    )
    nest_stft = torch.stft(
        nest_preemph, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        center=True, window=window.float(), return_complex=True, pad_mode="constant"
    )
    
    # Magnitude and power
    nemo_mag = torch.sqrt(torch.view_as_real(nemo_stft).pow(2).sum(-1))
    nest_mag = torch.sqrt(torch.view_as_real(nest_stft).pow(2).sum(-1))
    
    nemo_power = nemo_mag.pow(2.0)
    nest_power = nest_mag.pow(2.0)
    
    # Mel filterbank
    fb = nemo_fb.squeeze(0) if isinstance(nemo_fb, torch.Tensor) else None
    if fb is not None:
        nemo_mel = torch.matmul(fb, nemo_power)
        nest_mel = torch.matmul(fb, nest_power)
    else:
        print("   No filterbank available")
        return
    
    # Log
    nemo_log_mel = torch.log(nemo_mel + log_zero_guard)
    nest_log_mel = torch.log(nest_mel + log_zero_guard)
    
    print(f"   Log Mel shape: {nemo_log_mel.shape}")
    
    log_mel_diff = (nemo_log_mel - nest_log_mel).abs()
    print(f"   Log Mel max diff: {log_mel_diff.max().item():.6e}")
    
    # Compute seq_len for features
    pad_amount = n_fft // 2 * 2
    nemo_feat_len = torch.floor_divide((nemo_audio_len + pad_amount - n_fft), hop_length) + 1
    nest_feat_len = torch.floor_divide((nest_audio_len + pad_amount - n_fft), hop_length) + 1
    
    print(f"\n4. Feature seq_len:")
    print(f"   NeMo: {nemo_feat_len.tolist()}")
    print(f"   nest: {nest_feat_len.tolist()}")
    
    # Now run normalize_batch with debug
    print(f"\n5. Running normalize_batch with debug...")
    
    nemo_result, nemo_mean, nemo_std, nemo_debug = normalize_batch_debug(
        nemo_log_mel, nemo_feat_len, "per_feature"
    )
    nest_result, nest_mean, nest_std, nest_debug = normalize_batch_debug(
        nest_log_mel, nest_feat_len, "per_feature"
    )
    
    # Compare intermediate values
    print(f"\n6. Comparing intermediate values:")
    
    print(f"\n   valid_mask_sum (should be seq_len):")
    print(f"     NeMo: {nemo_debug['valid_mask_sum'].tolist()}")
    print(f"     nest: {nest_debug['valid_mask_sum'].tolist()}")
    
    print(f"\n   x_mean_denominator:")
    print(f"     NeMo: {nemo_debug['x_mean_denominator'].tolist()}")
    print(f"     nest: {nest_debug['x_mean_denominator'].tolist()}")
    
    # Compare x_mean
    mean_diff = (nemo_debug['x_mean'] - nest_debug['x_mean']).abs()
    print(f"\n   x_mean max diff: {mean_diff.max().item():.6e}")
    
    # Compare x_std
    std_diff = (nemo_debug['x_std'] - nest_debug['x_std']).abs()
    print(f"   x_std max diff: {std_diff.max().item():.6e}")
    
    # Compare result
    result_diff = (nemo_result - nest_result).abs()
    print(f"\n   Normalized result max diff: {result_diff.max().item():.6e}")
    
    # Compare with actual featurizer output
    print(f"\n7. Loading actual featurizer outputs...")
    
    try:
        with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
            nemo_layers = pickle.load(f)
        with open(nest_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
            nest_layers = pickle.load(f)
        
        nemo_feat = nemo_layers.get('preprocessor.featurizer') or nemo_layers.get('preprocessor')
        nest_feat = nest_layers.get('preprocessor.featurizer') or nest_layers.get('preprocessor')
        
        nemo_actual = nemo_feat['forward_outputs'][0]
        nest_actual = nest_feat['forward_outputs'][0]
        
        print(f"   Actual output shapes: NeMo={nemo_actual.shape}, nest={nest_actual.shape}")
        
        # Compare manual vs actual
        # Need to handle padding
        manual_len = nemo_result.shape[2]
        actual_len = nemo_actual.shape[2]
        
        print(f"   Manual length: {manual_len}, Actual length: {actual_len}")
        
        if manual_len <= actual_len:
            nemo_manual_vs_actual = (nemo_result - nemo_actual[:, :, :manual_len]).abs()
            nest_manual_vs_actual = (nest_result - nest_actual[:, :, :manual_len]).abs()
            
            print(f"\n   Manual vs Actual NeMo: max diff = {nemo_manual_vs_actual.max().item():.6e}")
            print(f"   Manual vs Actual nest: max diff = {nest_manual_vs_actual.max().item():.6e}")
            
            # Check first frame specifically
            print(f"\n8. First Frame Analysis:")
            print(f"   Manual NeMo[0, :5, 0]: {nemo_result[0, :5, 0].tolist()}")
            print(f"   Actual NeMo[0, :5, 0]: {nemo_actual[0, :5, 0].tolist()}")
            print(f"   Manual nest[0, :5, 0]: {nest_result[0, :5, 0].tolist()}")
            print(f"   Actual nest[0, :5, 0]: {nest_actual[0, :5, 0].tolist()}")
            
    except Exception as e:
        print(f"   Error loading layer outputs: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Diagnosis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

