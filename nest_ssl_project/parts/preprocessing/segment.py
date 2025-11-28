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
    
    def __init__(self, samples: torch.Tensor, sample_rate: int):
        """
        Initialize audio segment.
        
        Args:
            samples: Audio samples as torch tensor
            sample_rate: Sample rate in Hz
        """
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
        
        # Load audio
        try:
            # Try soundfile first (faster, especially for seeking)
            with sf.SoundFile(str(audio_file)) as sf_file:
                sr = sf_file.samplerate
                if duration is not None:
                    num_frames = int(duration * sr)
                else:
                    num_frames = -1
                
                if offset > 0:
                    sf_file.seek(int(offset * sr))
                
                # Optimize: read directly as numpy array, then convert to torch once
                samples = sf_file.read(frames=num_frames, dtype='float32')
        except Exception:
            # Fallback to librosa
            samples, sr = librosa.load(
                str(audio_file),
                sr=None,
                offset=offset,
                duration=duration,
            )
        
        # Resample if needed (optimize: only resample if necessary)
        if target_sr is not None and target_sr != sr:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Optimize: convert numpy array to torch tensor once (avoid double conversion)
        # Use from_numpy for better performance (shares memory if possible)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).float()
        else:
            samples = torch.tensor(samples, dtype=torch.float32)
        
        return cls(samples=samples, sample_rate=sr)


# Export available formats from soundfile
available_formats = sf.available_formats()

# Channel selector type
ChannelSelectorType = Union[int, Iterable[int], str]

