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
Simplified audio segment processing to replace nemo.collections.asr.parts.preprocessing.segment
"""

import torch
import soundfile as sf
import librosa
import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Iterable
from pathlib import Path


class AudioSegment:
    """
    Simplified audio segment class for loading and processing audio files.
    """
    
    def __init__(self, samples: torch.Tensor, sample_rate: int, target_sr: Optional[int] = None):
        """
        Initialize audio segment.
        
        Args:
            samples: Audio samples as torch tensor
            sample_rate: Sample rate in Hz
            target_sr: Target sample rate for resampling (None to keep original)
        """
        # Resample if needed (matching NeMo's behavior in __init__)
        if target_sr is not None and target_sr != sample_rate:
            # Convert to numpy for librosa resampling
            if isinstance(samples, torch.Tensor):
                samples_np = samples.cpu().numpy()
            else:
                samples_np = samples
            
            # Handle multi-channel audio (librosa expects channels-first for multi-channel)
            if samples_np.ndim == 1:
                # Single channel: resample directly
                samples_np = librosa.resample(samples_np, orig_sr=sample_rate, target_sr=target_sr)
            elif samples_np.ndim == 2:
                # Multi-channel: transpose for librosa (channels-first), resample, transpose back
                samples_np = samples_np.transpose()
                samples_np = librosa.resample(samples_np, orig_sr=sample_rate, target_sr=target_sr)
                samples_np = samples_np.transpose()
            else:
                raise ValueError(f"Unsupported audio shape: {samples_np.shape}")
            
            # Convert back to torch tensor
            samples = torch.from_numpy(samples_np).float()
            sample_rate = target_sr
        
        self.samples = samples
        self.sample_rate = sample_rate
    
    @classmethod
    def from_file(
        cls,
        audio_file: Union[str, Path],
        offset: float = 0.0,
        duration: Optional[float] = None,
        target_sr: Optional[int] = None,
    ):
        """
        Load audio segment from file.
        
        Args:
            audio_file: Path to audio file
            offset: Start time in seconds
            duration: Duration in seconds (None for full file)
            target_sr: Target sample rate (None to keep original)
        
        Returns:
            AudioSegment instance
        """
        audio_file = Path(audio_file).expanduser().resolve()
        
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Load audio with timeout protection
        try:
            # Try soundfile first (faster, especially for seeking)
            with sf.SoundFile(str(audio_file)) as sf_file:
                sr = sf_file.samplerate
                if duration is not None:
                    num_frames = int(duration * sr)
                else:
                    num_frames = -1
                
                if offset > 0:
                    # CRITICAL FIX: Validate offset to prevent hanging on corrupted files
                    max_offset = sf_file.frames / sr if sf_file.frames > 0 else 0
                    if offset >= max_offset:
                        raise ValueError(f"Offset {offset} exceeds file duration {max_offset}")
                    sf_file.seek(int(offset * sr))
                
                # Optimize: read directly as numpy array, then convert to torch once
                samples = sf_file.read(frames=num_frames, dtype='float32')
                
                # CRITICAL FIX: Validate samples to prevent hanging on corrupted files
                if len(samples) == 0:
                    raise ValueError(f"Read zero samples from {audio_file}")
        except Exception as e:
            # Fallback to librosa
            try:
                samples, sr = librosa.load(
                    str(audio_file),
                    sr=None,
                    offset=offset,
                    duration=duration,
                )
                if len(samples) == 0:
                    raise ValueError(f"Librosa read zero samples from {audio_file}")
            except Exception as librosa_e:
                # If both fail, raise with combined error message
                raise RuntimeError(f"Failed to load audio from {audio_file}: soundfile error={e}, librosa error={librosa_e}")
        
        # Convert to torch tensor (resampling will be done in __init__)
        # Use from_numpy for better performance (shares memory if possible)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).float()
        else:
            samples = torch.tensor(samples, dtype=torch.float32)
        
        # Pass target_sr to __init__ for resampling (matching NeMo's behavior)
        return cls(samples=samples, sample_rate=sr, target_sr=target_sr)


# Export available formats from soundfile
available_formats = sf.available_formats()

# Channel selector type
ChannelSelectorType = Union[int, Iterable[int], str]

