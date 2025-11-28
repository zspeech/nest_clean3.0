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

from dataclasses import dataclass


@dataclass
class CacheAwareStreamingConfig:
    chunk_size: int = (
        0  # the size of each chunk at each step, it can be a list of two integers to specify different chunk sizes for the first step and others
    )
    shift_size: int = (
        0  # the size of the shift in each step, it can be a list of two integers to specify different shift sizes for the first step and others
    )

    cache_drop_size: int = 0  # the number of steps to drop from the cache
    last_channel_cache_size: int = 0  # the size of the needed cache for last channel layers

    valid_out_len: int = (
        0  # the number of the steps in the final output which are valid (have the same value as in the offline mode)
    )

    pre_encode_cache_size: int = (
        0  # the size of the needed cache for the pre-encoding part of the model to avoid caching inside the pre-encoding layers
    )
    drop_extra_pre_encoded: int = 0  # the number of steps to get dropped after the pre-encoding layer

    last_channel_num: int = 0  # number of the last channel layers (like MHA layers) which need caching in the model
    last_time_num: int = 0  # number of the last time layers (like convolutions) which need caching in the model

