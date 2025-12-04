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
Simplified common utilities to replace nemo.core.classes.common
"""

from functools import wraps
from typing import Optional, Dict, Any, Callable
from abc import ABC

from core.neural_types import NeuralType


def typecheck():
    """
    Decorator for type checking (simplified - no-op for now).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


class Typing(ABC):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None
    
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None


class PretrainedModelInfo:
    """Information about a pretrained model."""
    def __init__(self, pretrained_model_name: str, description: str, location: str):
        self.pretrained_model_name = pretrained_model_name
        self.description = description
        self.location = location
