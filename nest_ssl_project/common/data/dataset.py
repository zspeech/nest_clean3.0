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
Simplified ConcatDataset to replace nemo.collections.common.data.dataset.ConcatDataset
"""

import numpy as np
from typing import List, Any, Optional, Iterator
from torch.utils.data import IterableDataset, Dataset


class ConcatDataset(IterableDataset):
    """
    A dataset that accepts multiple datasets and samples from them based on the specified
    sampling technique.
    
    Simplified version of NeMo's ConcatDataset.
    """
    
    def __init__(
        self,
        datasets: List[Any],
        shuffle: bool = True,
        sampling_technique: str = 'temperature',
        sampling_temperature: int = 5,
        sampling_scale: int = 1,
        sampling_probabilities: List[float] = None,
        seed: Optional[int] = None,
        global_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        
        self.datasets = datasets
        self.shuffle = shuffle
        self.global_rank = global_rank
        self.world_size = world_size
        self.sampling_scale = sampling_scale
        
        # Determine dataset type
        if len(datasets) > 0:
            if isinstance(datasets[0], IterableDataset):
                self.kind = 'iterable'
            else:
                self.kind = 'map'
        else:
            self.kind = 'map'
        
        # Setup sampling
        supported_techniques = ['temperature', 'random', 'round-robin']
        if sampling_technique not in supported_techniques:
            raise ValueError(f"Sampling technique must be one of {supported_techniques}")
        
        self.sampling_technique = sampling_technique
        self.sampling_temperature = sampling_temperature
        self.sampling_probabilities = sampling_probabilities
        self.np_rng = np.random.RandomState(seed)
        
        # Calculate total length
        if self.kind == 'map':
            self.length = sum(len(d) for d in datasets) * sampling_scale
        else:
            # For iterable datasets, length is approximate
            self.length = 0
    
    def __iter__(self) -> Iterator:
        """Iterate over datasets based on sampling technique."""
        if self.kind == 'iterable':
            # For iterable datasets, use round-robin by default
            iterators = [iter(d) for d in self.datasets]
            exhausted = [False] * len(self.datasets)
            
            while not all(exhausted):
                if self.sampling_technique == 'round-robin':
                    for i, it in enumerate(iterators):
                        if not exhausted[i]:
                            try:
                                yield next(it)
                            except StopIteration:
                                exhausted[i] = True
                elif self.sampling_technique == 'random':
                    # Random sampling for iterable datasets
                    active_indices = [i for i, ex in enumerate(exhausted) if not ex]
                    if active_indices:
                        idx = self.np_rng.choice(active_indices)
                        try:
                            yield next(iterators[idx])
                        except StopIteration:
                            exhausted[idx] = True
                else:
                    # Temperature sampling
                    active_indices = [i for i, ex in enumerate(exhausted) if not ex]
                    if active_indices:
                        # Simple temperature-based selection
                        weights = [1.0 / self.sampling_temperature] * len(active_indices)
                        idx = self.np_rng.choice(active_indices, p=np.array(weights) / sum(weights))
                        try:
                            yield next(iterators[idx])
                        except StopIteration:
                            exhausted[idx] = True
        else:
            # For map-style datasets
            all_indices = []
            for dataset_idx, dataset in enumerate(self.datasets):
                indices = list(range(len(dataset)))
                if self.shuffle:
                    self.np_rng.shuffle(indices)
                all_indices.extend([(dataset_idx, idx) for idx in indices])
            
            # Apply sampling scale
            all_indices = all_indices * self.sampling_scale
            
            if self.shuffle:
                self.np_rng.shuffle(all_indices)
            
            for dataset_idx, sample_idx in all_indices:
                yield self.datasets[dataset_idx][sample_idx]
    
    def __len__(self):
        """Return approximate length."""
        return self.length

