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
Simplified config utilities to replace nemo.core.config
"""

from omegaconf import DictConfig, OmegaConf
from typing import Callable, Any
import functools


def hydra_runner(config_path: str = None, config_name: str = None):
    """
    Decorator for Hydra-based configuration management.
    Simplified version that loads config and passes it to the function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If config_path and config_name are provided, load the config
            if config_path and config_name:
                config_file = f"{config_path}/{config_name}.yaml"
                cfg = OmegaConf.load(config_file)
                
                # Merge with command-line overrides from kwargs
                if kwargs:
                    cfg = OmegaConf.merge(cfg, OmegaConf.create(kwargs))
                
                return func(cfg)
            else:
                # No config file, just call the function
                return func(*args, **kwargs)
        return wrapper
    return decorator

