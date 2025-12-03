#!/usr/bin/env python3
"""
Compare featurizer step by step between NeMo and nest.
Run NeMo featurizer and nest featurizer on same input, compare each step.
"""

import torch
import torch.nn.functional as F
import pickle
from pathlib import Path
import argparse
import sys

# Try to import NeMo
try:
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures, normalize_batch
    HAVE_NEMO = True
except ImportError:
    HAVE_NEMO = False
    print("NeMo not available.")

# Try to import nest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from nest_ssl_project.modules.audio_preprocessing import FilterbankFeatures as NestFilterbankFeatures
    from nest_ssl_project.modules.audio_preprocessing import normalize_batch as nest_normalize_batch
    HAVE_NEST = True
except ImportError:
    HAVE_NEST = False
    print("nest not available.")


def compare_tensors(name, t1, t2, label1="NeMo", label2="nest"):
    """Compare two tensors and print results."""
    if t1.shape != t2.shape:
        print(f"  {name}: SHAPE MISMATCH - {label1}={t1.shape}, {label2}={t2.shape}")
        return False
    
    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    if max_diff < 1e-5:
        status = "[OK]"
    elif max_diff < 1e-3:
        status = "[WARN]"
    else:
        status = "[FAIL]"
    
    print(f"  {name}: {status} max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    
    if max_diff > 1e-4:
        # Find location of max diff
        max_idx = diff.argmax()
        flat_idx = max_idx.item()
        # Convert to multi-dimensional index
        idx = []
        for dim in reversed(t1.shape):
            idx.insert(0, flat_idx % dim)
            flat_idx //= dim
        idx = tuple(idx)
        print(f"       Max diff at {idx}: {label1}={t1[idx].item():.8f}, {label2}={t2[idx].item():.8f}")
    
    return max_diff < 1e-4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    
    if not HAVE_NEMO or not HAVE_NEST:
        print("Both NeMo and nest are required.")
        return
    
    nemo_dir = Path(args.nemo_dir)
    
    if args.step is None:
        nemo_steps = [int(d.name.split('_')[1]) for d in nemo_dir.glob("step_*") if d.is_dir()]
        if nemo_steps:
            args.step = sorted(nemo_steps)[0]
    
    print("="*80)
    print("COMPARE STEP BY STEP: NeMo vs nest")
    print("="*80)
    
    # Load batch
    batch = torch.load(nemo_dir / f"step_{args.step}" / "batch.pt", weights_only=False)
    audio = batch['audio']
    audio_len = batch['audio_len']
    
    # Load buffers
    buffers = torch.load(nemo_dir / "buffers" / "buffers.pt", weights_only=False)
    
    print(f"\n1. Input:")
    print(f"   Audio shape: {audio.shape}")
    print(f"   Audio len: {audio_len.tolist()}")
    
    # Create NeMo featurizer
    nemo_feat = FilterbankFeatures(
        sample_rate=16000,
        n_window_size=400,
        n_window_stride=160,
        n_fft=512,
        nfilt=80,
        lowfreq=0.0,
        highfreq=8000.0,
        window="hann",
        normalize="per_feature",
        preemph=0.97,
        dither=0.0,
        pad_to=16,
        mag_power=2.0,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        exact_pad=False,
    )
    
    # Create nest featurizer
    nest_feat = NestFilterbankFeatures(
        sample_rate=16000,
        n_window_size=400,
        n_window_stride=160,
        n_fft=512,
        nfilt=80,
        lowfreq=0.0,
        highfreq=8000.0,
        window="hann",
        normalize="per_feature",
        preemph=0.97,
        dither=0.0,
        pad_to=16,
        mag_power=2.0,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        exact_pad=False,
    )
    
    # Copy buffers
    fb_key = 'preprocessor.featurizer.fb' if 'preprocessor.featurizer.fb' in buffers else 'preprocessor.fb'
    window_key = 'preprocessor.featurizer.window' if 'preprocessor.featurizer.window' in buffers else 'preprocessor.window'
    
    if fb_key in buffers:
        nemo_feat.fb.data = buffers[fb_key]
        nest_feat.fb.data = buffers[fb_key]
    if window_key in buffers:
        nemo_feat.window.data = buffers[window_key]
        nest_feat.window.data = buffers[window_key]
    
    print(f"\n2. Featurizer buffers:")
    compare_tensors("fb", nemo_feat.fb, nest_feat.fb)
    compare_tensors("window", nemo_feat.window, nest_feat.window)
    
    # Set both to eval mode
    nemo_feat.eval()
    nest_feat.eval()
    
    print(f"\n3. Step-by-step comparison:")
    
    # Get seq_len
    nemo_seq_len = nemo_feat.get_seq_len(audio_len)
    nest_seq_len = nest_feat.get_seq_len(audio_len)
    print(f"\n   get_seq_len:")
    print(f"     NeMo: {nemo_seq_len.tolist()}")
    print(f"     nest: {nest_seq_len.tolist()}")
    compare_tensors("seq_len", nemo_seq_len, nest_seq_len)
    
    # Preemphasis
    x_nemo = audio.clone()
    x_nest = audio.clone()
    seq_len_time = audio_len.clone()
    
    # NeMo preemphasis
    timemask_nemo = torch.arange(x_nemo.shape[1], device=x_nemo.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
    x_nemo = torch.cat((x_nemo[:, 0].unsqueeze(1), x_nemo[:, 1:] - nemo_feat.preemph * x_nemo[:, :-1]), dim=1)
    x_nemo = x_nemo.masked_fill(~timemask_nemo, 0.0)
    
    # nest preemphasis (should be identical)
    timemask_nest = torch.arange(x_nest.shape[1], device=x_nest.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
    x_nest = torch.cat((x_nest[:, 0].unsqueeze(1), x_nest[:, 1:] - nest_feat.preemph * x_nest[:, :-1]), dim=1)
    x_nest = x_nest.masked_fill(~timemask_nest, 0.0)
    
    print(f"\n   After preemphasis:")
    compare_tensors("preemph", x_nemo, x_nest)
    
    # STFT
    x_stft_nemo = nemo_feat.stft(x_nemo)
    x_stft_nest = nest_feat.stft(x_nest)
    
    print(f"\n   After STFT:")
    print(f"     NeMo shape: {x_stft_nemo.shape}")
    print(f"     nest shape: {x_stft_nest.shape}")
    compare_tensors("stft_real", x_stft_nemo.real, x_stft_nest.real)
    compare_tensors("stft_imag", x_stft_nemo.imag, x_stft_nest.imag)
    
    # Magnitude
    x_real_nemo = torch.view_as_real(x_stft_nemo)
    x_mag_nemo = torch.sqrt(x_real_nemo.pow(2).sum(-1))
    
    x_real_nest = torch.view_as_real(x_stft_nest)
    x_mag_nest = torch.sqrt(x_real_nest.pow(2).sum(-1))
    
    print(f"\n   After magnitude:")
    compare_tensors("magnitude", x_mag_nemo, x_mag_nest)
    
    # Power
    x_pow_nemo = x_mag_nemo.pow(2.0)
    x_pow_nest = x_mag_nest.pow(2.0)
    
    print(f"\n   After power:")
    compare_tensors("power", x_pow_nemo, x_pow_nest)
    
    # Mel filterbank
    x_mel_nemo = torch.matmul(nemo_feat.fb.to(x_pow_nemo.dtype), x_pow_nemo)
    x_mel_nest = torch.matmul(nest_feat.fb.to(x_pow_nest.dtype), x_pow_nest)
    
    print(f"\n   After mel filterbank:")
    compare_tensors("mel", x_mel_nemo, x_mel_nest)
    
    # Log
    log_guard = 2**-24
    x_log_nemo = torch.log(x_mel_nemo + log_guard)
    x_log_nest = torch.log(x_mel_nest + log_guard)
    
    print(f"\n   After log:")
    compare_tensors("log", x_log_nemo, x_log_nest)
    
    # Normalize
    x_norm_nemo, mean_nemo, std_nemo = normalize_batch(x_log_nemo, nemo_seq_len, normalize_type="per_feature")
    x_norm_nest, mean_nest, std_nest = nest_normalize_batch(x_log_nest, nest_seq_len, normalize_type="per_feature")
    
    print(f"\n   After normalize:")
    compare_tensors("norm", x_norm_nemo, x_norm_nest)
    compare_tensors("mean", mean_nemo, mean_nest)
    compare_tensors("std", std_nemo, std_nest)
    
    # Full forward pass
    print(f"\n4. Full forward pass comparison:")
    with torch.no_grad():
        out_nemo, len_nemo = nemo_feat(audio, audio_len)
        out_nest, len_nest = nest_feat(audio, audio_len)
    
    print(f"   NeMo output: shape={out_nemo.shape}, len={len_nemo.tolist()}")
    print(f"   nest output: shape={out_nest.shape}, len={len_nest.tolist()}")
    compare_tensors("full_output", out_nemo, out_nest)
    compare_tensors("full_len", len_nemo, len_nest)
    
    # Compare with saved
    print(f"\n5. Compare with saved outputs:")
    with open(nemo_dir / f"step_{args.step}" / "layer_outputs.pkl", 'rb') as f:
        layers = pickle.load(f)
    
    saved_out = layers.get('preprocessor.featurizer', {}).get('forward_outputs', [])[0]
    saved_len = layers.get('preprocessor.featurizer', {}).get('forward_outputs', [])[1]
    
    print(f"   Saved output: shape={saved_out.shape}, len={saved_len.tolist()}")
    compare_tensors("nemo_vs_saved", out_nemo, saved_out, "NeMo_new", "NeMo_saved")
    compare_tensors("nest_vs_saved", out_nest, saved_out, "nest", "NeMo_saved")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

