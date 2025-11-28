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


from collections import OrderedDict

import torch

import torch
from core.classes.neural_module import NeuralModule
from core.classes.common import typecheck
from core.neural_types import AcousticEncodedRepresentation, LogprobsType, NeuralType

# Simple weight initialization function
def init_weights(m, mode='xavier_uniform'):
    """Initialize weights for a module."""
    if isinstance(m, torch.nn.Linear):
        if mode == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight)
        elif mode == 'xavier_normal':
            torch.nn.init.xavier_normal_(m.weight)
        elif mode == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(m.weight)
        elif mode == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        if mode == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight)
        elif mode == 'xavier_normal':
            torch.nn.init.xavier_normal_(m.weight)
        elif mode == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(m.weight)
        elif mode == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class MultiSoftmaxDecoder(NeuralModule):
    """
    A linear decoder that takes encoder output and produces log probabilities, which also handles the
    case where each frame has multiple output targets. This can be used together with
    `nemo.collections.asr.losses.ssl_losses.MultiMLMLoss` to train a model with multiple targets per frame.
    """

    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        if self.squeeze_single and self.num_decoders == 1:
            return OrderedDict({"logprobs": NeuralType(('B', 'T', 'C'), LogprobsType())})
        return OrderedDict({"logprobs": NeuralType(('B', 'T', 'C', 'H'), LogprobsType())})

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        num_decoders: int = 1,
        init_mode: str = "xavier_uniform",
        use_bias: bool = False,
        squeeze_single: bool = False,
    ) -> None:
        """
        Args:
            feat_in: input feature dimension
            num_classes: number of classes
            num_decoders: number of decoders
            init_mode: initialization mode
            use_bias: whether to use bias
            squeeze_single: if True, squeeze codebook dimension if num_books is 1
        """
        super().__init__()
        self.feat_in = feat_in
        self.num_classes = num_classes
        self.num_decoders = num_decoders
        self.squeeze_single = squeeze_single

        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self.feat_in, self.num_classes * self.num_decoders, kernel_size=1, bias=use_bias)
        )
        self.apply(lambda x: init_weights(x, mode=init_mode))

    @typecheck()
    def forward(self, encoder_output):
        """
        Args:
            encoder_output: output from the encoder of shape (B, D, T)
        Returns:
            logprobs: log probabilities of shape (B, T, C, H), or (B, T, C) if squeeze_single is True
        """
        logits = self.decoder_layers(encoder_output).transpose(1, 2)
        logits = logits.reshape(logits.shape[0], logits.shape[1], self.num_classes, self.num_decoders)
        if self.squeeze_single and self.num_decoders == 1:
            logits = logits.squeeze(-1)

        return torch.nn.functional.log_softmax(logits, dim=2)
