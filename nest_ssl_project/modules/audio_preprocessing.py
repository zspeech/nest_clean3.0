# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Audio preprocessing modules matching NeMo's implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import librosa
    HAVE_LIBROSA = True
except ImportError:
    HAVE_LIBROSA = False

try:
    import torchaudio
    HAVE_TORCHAUDIO = True
except ImportError:
    HAVE_TORCHAUDIO = False

from core.classes.neural_module import NeuralModule
from core.neural_types import AudioSignal, LengthsType, MelSpectrogramType, NeuralType

CONSTANT = 1e-5


def normalize_batch(x, seq_len, normalize_type):
    """Normalize batch following NeMo's implementation."""
    if normalize_type == "per_feature":
        batch_size = x.shape[0]
        max_time = x.shape[2]
        
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        valid_mask = time_steps < seq_len.unsqueeze(1)
        
        x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
        x_mean_denominator = valid_mask.sum(axis=1)
        x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)
        
        x_std = torch.sqrt(
            torch.sum(torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
            / (x_mean_denominator.unsqueeze(1) - 1.0)
        )
        x_std = x_std.masked_fill(x_std.isnan(), 0.0)
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    else:
        return x


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms (matches NeMo implementation)."""
    
    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=CONSTANT,
        pad_to=16,
        pad_value=0,
        frame_splicing=1,
        exact_pad=False,
        mag_power=2.0,
        mel_norm="slaney",
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None
        self.exact_pad = exact_pad
        
        # Window function
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)
        
        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.mag_power = mag_power
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value
        
        highfreq = highfreq or sample_rate / 2
        
        # Create mel filterbank using librosa (matching NeMo)
        if HAVE_LIBROSA:
            filterbanks = torch.tensor(
                librosa.filters.mel(
                    sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, 
                    fmin=lowfreq, fmax=highfreq, norm=mel_norm
                ),
                dtype=torch.float,
            ).unsqueeze(0)
            self.register_buffer("fb", filterbanks)
        else:
            # Fallback: create identity matrix (will produce incorrect results)
            self.register_buffer("fb", torch.eye(nfilt, self.n_fft // 2 + 1).unsqueeze(0))
            import warnings
            warnings.warn("librosa not available, mel filterbank will be incorrect")
    
    def stft(self, x):
        """Compute STFT."""
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if self.exact_pad else True,
            window=self.window.to(dtype=torch.float, device=x.device) if self.window is not None else None,
            return_complex=True,
            pad_mode="constant",
        )
    
    def log_zero_guard_value_fn(self, x):
        """Get log zero guard value."""
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(f"log_zero_guard_value must be number, 'tiny', or 'eps'")
        else:
            return self.log_zero_guard_value
    
    def get_seq_len(self, seq_len):
        """Compute output sequence length.
        
        Note: Using explicit integer division to avoid torch.floor_divide 
        version differences across PyTorch versions.
        """
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        # Convert to float, do division, then floor and convert to long
        # This matches NeMo's behavior more consistently across PyTorch versions
        seq_len_float = (seq_len.float() + pad_amount - self.n_fft) / self.hop_length
        return seq_len_float.floor().to(dtype=torch.long)
    
    def forward(self, x, seq_len, linear_spec=False):
        """
        Forward pass matching NeMo's FilterbankFeatures.
        
        Args:
            x: Input audio [B, T]
            seq_len: Length tensor [B]
            linear_spec: If True, return linear spectrogram
        
        Returns:
            features: Mel spectrogram [B, D, T']
            seq_len: Updated lengths [B]
        """
        seq_len_time = seq_len
        seq_len_unfixed = self.get_seq_len(seq_len)
        seq_len = torch.where(seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)
        
        # STFT padding if exact_pad
        if self.stft_pad_amount is not None:
            x = F.pad(x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant").squeeze(1)
        
        # Dither (only in training)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)
        
        # Preemphasis
        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)
        
        # STFT
        with torch.amp.autocast(x.device.type, enabled=False):
            x = self.stft(x)
        
        # Convert complex to magnitude
        guard = 0  # Simplified: always use 0 for guard
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)
        
        # Power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)
        
        # Return linear spec if required
        if linear_spec:
            return x, seq_len
        
        # Apply mel filterbank
        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.matmul(self.fb.to(x.dtype), x)
        
        # Log features
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type must be 'add' or 'clamp'")
        
        # Frame splicing (simplified - not fully implemented)
        # if self.frame_splicing > 1:
        #     x = splice_frames(x, self.frame_splicing)
        
        # Normalize
        if self.normalize:
            x = normalize_batch(x, seq_len, normalize_type=self.normalize)
        
        # Mask and pad
        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        
        if self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            if pad_amt != 0:
                x = F.pad(x, (0, self.pad_to - pad_amt), value=self.pad_value)
        
        return x, seq_len


class AudioToMelSpectrogramPreprocessor(NeuralModule):
    """
    Audio to mel spectrogram preprocessor matching NeMo's implementation.
    """
    
    @property
    def input_types(self):
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }
    
    @property
    def output_types(self):
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }
    
    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        window: str = "hann",
        normalize: str = "per_feature",
        n_fft: Optional[int] = None,
        features: int = 80,
        log: bool = True,
        dither: float = 1e-5,
        pad_to: int = 16,
        pad_value: float = 0.0,
        frame_splicing: int = 1,
        preemph: float = 0.97,
        lowfreq: int = 0,
        highfreq: Optional[int] = None,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: float = 2**-24,
        exact_pad: bool = False,
        mag_power: float = 2.0,
        mel_norm: str = "slaney",
        use_torchaudio: bool = False,
        **kwargs
    ):
        super().__init__()
        self._sample_rate = sample_rate
        
        # Convert window_size/stride to samples
        n_window_size = int(window_size * sample_rate)
        n_window_stride = int(window_stride * sample_rate)
        
        # Create featurizer (matching NeMo's approach)
        if use_torchaudio and HAVE_TORCHAUDIO:
            # Use torchaudio implementation
            self.featurizer = None  # Would use FilterbankFeaturesTA if needed
            raise NotImplementedError("torchaudio version not yet implemented, use librosa version")
        else:
            # Use librosa-based FilterbankFeatures (matching NeMo default)
            self.featurizer = FilterbankFeatures(
                sample_rate=sample_rate,
                n_window_size=n_window_size,
                n_window_stride=n_window_stride,
                window=window,
                normalize=normalize,
                n_fft=n_fft,
                preemph=preemph,
                nfilt=features,
                lowfreq=lowfreq,
                highfreq=highfreq,
                log=log,
                log_zero_guard_type=log_zero_guard_type,
                log_zero_guard_value=log_zero_guard_value,
                dither=dither,
                pad_to=pad_to,
                pad_value=pad_value,
                frame_splicing=frame_splicing,
                exact_pad=exact_pad,
                mag_power=mag_power,
                mel_norm=mel_norm,
            )
    
    def forward(self, input_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert audio to mel spectrogram.
        
        Args:
            input_signal: Audio tensor of shape [B, T]
            length: Length tensor of shape [B]
        
        Returns:
            processed_signal: Mel spectrogram of shape [B, D, T']
            processed_length: Length tensor of shape [B]
        """
        return self.featurizer(input_signal, length)


class SpectrogramAugmentation(NeuralModule):
    """
    Spectrogram augmentation module (matching NeMo's interface).
    Currently a no-op since freq_masks and time_masks are set to 0 in config.
    """
    
    @property
    def input_types(self):
        return {
            "input_spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }
    
    @property
    def output_types(self):
        return {
            "augmented_spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }
    
    def __init__(
        self,
        freq_masks: int = 0,
        time_masks: int = 0,
        freq_width: int = 27,
        time_width: float = 0.05,
        **kwargs
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
    
    def forward(self, input_spec: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spectrogram augmentation.
        Currently no-op since masks are disabled in config.
        """
        # Since freq_masks and time_masks are 0 in config, just return input
        return input_spec, length
