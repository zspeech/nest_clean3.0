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
Simplified neural types to replace nemo.core.neural_types
These are mostly type hints for documentation and type checking.
"""

from typing import Any, Optional, Tuple


class ElementType:
    """Base class for element types."""
    pass


class AudioSignal(ElementType):
    """Audio signal element type."""
    def __init__(self, freq: Optional[int] = None):
        self.freq = freq


class SpectrogramType(ElementType):
    """Spectrogram element type."""
    pass


class MelSpectrogramType(SpectrogramType):
    """Mel spectrogram element type (alias for SpectrogramType)."""
    pass


class AcousticEncodedRepresentation(ElementType):
    """Acoustic encoded representation element type."""
    pass


class LabelsType(ElementType):
    """Labels element type."""
    pass


class LengthsType(ElementType):
    """Lengths element type."""
    pass


class ChannelType(ElementType):
    """Channel element type."""
    pass


class BoolType(ElementType):
    """Boolean element type."""
    pass


class LogprobsType(ElementType):
    """Log probabilities element type."""
    pass


class LossType(ElementType):
    """Loss element type."""
    pass


class AxisType:
    """Axis type for tensor dimensions."""
    def __init__(self, name: str):
        self.name = name


class NeuralType:
    """
    Neural type definition for tensor shapes and element types.
    Simplified version - mainly for documentation.
    """
    def __init__(
        self,
        axes: Optional[Tuple[str, ...]] = None,
        elements_type: Optional[ElementType] = None,
        optional: bool = False
    ):
        self.axes = axes
        self.elements_type = elements_type or ElementType()
        self.optional = optional
    
    def __call__(self, *args, **kwargs):
        """Allow calling as function for compatibility."""
        return self

