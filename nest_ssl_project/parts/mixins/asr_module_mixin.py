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
Simplified ASR module mixin to replace nemo.collections.asr.parts.mixins.ASRModuleMixin
"""

from abc import ABC
from typing import Dict, Any
from omegaconf import DictConfig, open_dict

from utils.logging import get_logger

logger = get_logger(__name__)


class ASRModuleMixin(ABC):
    """
    Mixin class for ASR modules.
    Provides common functionality for ASR models.
    """
    
    @staticmethod
    def inject_dataloader_value_from_model_config(
        model_cfg: Dict[str, Any], 
        dataloader_cfg: DictConfig, 
        key: str
    ):
        """
        Extracts a value from model config and propagates it to dataloader config.
        
        Args:
            model_cfg: Model configuration dictionary
            dataloader_cfg: Dataloader configuration (DictConfig)
            key: Key to extract from model_cfg and inject into dataloader_cfg
        """
        if key not in model_cfg:
            logger.info(
                f"Model level config does not contain `{key}`, "
                f"please explicitly provide `{key}` to the dataloaders."
            )
            return
        
        if not isinstance(dataloader_cfg, DictConfig):
            dataloader_cfg = DictConfig(dataloader_cfg)
        
        # If key exists in the data loader config
        if key in dataloader_cfg:
            # Dataloader key is provided and is non-null
            if dataloader_cfg[key] is not None and model_cfg[key] != dataloader_cfg[key]:
                # Model level key doesn't match Dataloader level key
                logger.warning(
                    f'`{key}` is explicitly provided to the data loader, and is different from '
                    f'the `{key}` provided at the model level config.\n'
                    f'If this is incorrect, please set the dataloader\'s `{key}` to None.'
                )
            else:
                # Dataloader key is None or values match - propagate from model level
                with open_dict(dataloader_cfg):
                    dataloader_cfg[key] = model_cfg[key]
        else:
            # If key doesn't exist in dataloader_cfg, inject it explicitly
            with open_dict(dataloader_cfg):
                dataloader_cfg[key] = model_cfg[key]

