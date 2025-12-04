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
Simplified NeuralModule base class to replace nemo.core.NeuralModule
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class NeuralModule(nn.Module, ABC):
    """
    Base class for all neural modules.
    This is a simplified version of NeMo's NeuralModule.
    """
    
    def __init__(self):
        super().__init__()
    
    @property
    def input_types(self) -> Optional[Dict[str, Any]]:
        """Returns definitions of module input ports."""
        return None
    
    @property
    def output_types(self) -> Optional[Dict[str, Any]]:
        """Returns definitions of module output ports."""
        return None
    
    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    

