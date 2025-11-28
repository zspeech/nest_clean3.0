# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import torch
import torch.nn as nn


class FusedBatchNorm1d(nn.Module):
    """
    Fused BatchNorm to use in Conformer to improve accuracy in finetuning with TTS scenario
    Drop-in replacement for BatchNorm1d with simple affine projection
    """

    def __init__(self, num_features: int):
        """
        Args:
            num_features: number of channels, see original BatchNorm1d documentation
        """
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.weight.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)

