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
Simplified audio featurization utilities to replace nemo.collections.asr.parts.preprocessing.features
"""

import torch
import numpy as np
from typing import Optional, Any
from parts.preprocessing.segment import AudioSegment
from parts.preprocessing.perturb import AudioAugmentor


class WaveformFeaturizer:
    """
    Featurizer for processing audio waveforms.
    """
    
    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        """
        Initialize waveform featurizer.
        
        Args:
            sample_rate: Target sample rate
            int_values: If True, load samples as 32-bit integers
            augmentor: Optional AudioAugmentor for augmentation
        """
        from parts.preprocessing.perturb import NoOpAugmentor
        self.augmentor = augmentor if augmentor is not None else NoOpAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values
    
    def max_augmentation_length(self, length):
        """Get maximum augmentation length."""
        if hasattr(self.augmentor, 'max_augmentation_length'):
            return self.augmentor.max_augmentation_length(length)
        return length
    
    def process(
        self,
        file_path,
        offset=0,
        duration=0,
        trim=False,
        trim_ref=np.max,
        trim_top_db=60,
        trim_frame_length=2048,
        trim_hop_length=512,
        orig_sr=None,
        channel_selector=None,
        normalize_db=None,
    ):
        """
        Process audio file and return features.
        
        Args:
            file_path: Path to audio file
            offset: Start offset in seconds
            duration: Duration in seconds (0 for full file)
            trim: Whether to trim silence
            trim_ref: Reference for trimming
            trim_top_db: Top dB for trimming
            trim_frame_length: Frame length for trimming
            trim_hop_length: Hop length for trimming
            orig_sr: Original sample rate
            channel_selector: Channel selector
            normalize_db: Normalization in dB
        
        Returns:
            torch.Tensor: Audio features
        """
        # Load audio segment
        audio = AudioSegment.from_file(
            file_path,
            offset=offset,
            duration=duration if duration > 0 else None,
            target_sr=self.sample_rate,
        )
        
        # Apply channel selection if needed
        if channel_selector is not None:
            audio.samples = self._select_channels(audio.samples, channel_selector)
        
        # Trim if requested
        if trim:
            audio.samples = self._trim_silence(
                audio.samples,
                top_db=trim_top_db,
                frame_length=trim_frame_length,
                hop_length=trim_hop_length,
            )
        
        # Normalize if requested
        if normalize_db is not None:
            audio.samples = self._normalize_db(audio.samples, normalize_db)
        
        # Convert to int if requested
        if self.int_values:
            audio.samples = (audio.samples * 2147483647).to(torch.int32)
        
        return self.process_segment(audio)
    
    def process_segment(self, audio_segment):
        """
        Process audio segment with augmentation.
        
        Args:
            audio_segment: AudioSegment instance
        
        Returns:
            torch.Tensor: Processed audio features
        """
        # Apply augmentation
        if self.augmentor is not None:
            self.augmentor.perturb(audio_segment)
        
        # Use detach().clone() to avoid warning about tensor construction
        if isinstance(audio_segment.samples, torch.Tensor):
            return audio_segment.samples.detach().clone().to(dtype=torch.float)
        else:
            return torch.tensor(audio_segment.samples, dtype=torch.float)
    
    def _select_channels(self, samples, channel_selector):
        """Select channels from multi-channel audio."""
        if samples.ndim == 1:
            return samples
        
        if isinstance(channel_selector, int):
            # Select single channel
            return samples[:, channel_selector] if samples.ndim > 1 else samples
        elif isinstance(channel_selector, str) and channel_selector == 'average':
            # Average all channels
            return samples.mean(dim=-1) if samples.ndim > 1 else samples
        elif hasattr(channel_selector, '__iter__'):
            # Select multiple channels
            return samples[:, list(channel_selector)]
        else:
            return samples
    
    def _trim_silence(self, samples, top_db=60, frame_length=2048, hop_length=512):
        """Trim silence from audio."""
        try:
            import librosa
            samples_np = samples.cpu().numpy()
            trimmed, _ = librosa.effects.trim(
                samples_np,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            return torch.tensor(trimmed, dtype=samples.dtype)
        except ImportError:
            # If librosa not available, return original
            return samples
    
    def _normalize_db(self, samples, target_db):
        """Normalize audio to target dB level."""
        # Convert to numpy for calculation
        samples_np = samples.cpu().numpy()
        
        # Calculate current RMS
        rms = np.sqrt(np.mean(samples_np ** 2))
        if rms > 0:
            # Calculate target RMS
            target_rms = 10 ** (target_db / 20.0)
            # Scale
            samples_np = samples_np * (target_rms / rms)
        
        return torch.tensor(samples_np, dtype=samples.dtype)
    
    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        """
        Create featurizer from config.
        
        Args:
            input_config: Input configuration dict
            perturbation_configs: Optional perturbation configs
        
        Returns:
            WaveformFeaturizer instance
        """
        from parts.preprocessing.perturb import process_augmentations
        
        if perturbation_configs is not None:
            augmentor = process_augmentations(perturbation_configs)
        else:
            augmentor = None
        
        sample_rate = input_config.get("sample_rate", 16000)
        int_values = input_config.get("int_values", False)
        
        return cls(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)

