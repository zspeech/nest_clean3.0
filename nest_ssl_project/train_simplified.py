#!/usr/bin/env python3
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
Training script using SimplifiedSSLModel.
This script trains the simplified SSL model which is aligned with the original EncDecDenoiseMaskedTokenPredModel.

Usage:
```sh
python train_simplified.py \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.train_ds.noise_manifest=<path to noise manifest> \
    model.validation_ds.manifest_filepath=<path to val manifest> \
    model.validation_ds.noise_manifest=<path to noise manifest> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=100
```
"""

import lightning.pytorch as pl
from omegaconf import OmegaConf

# Import local utilities
from utils.hydra_runner import hydra_runner
from utils.logging import get_logger, is_global_rank_zero
from utils.exp_manager import exp_manager

# Import simplified model
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ssl_models_simplified import SimplifiedSSLModel


logger = get_logger(__name__)


@hydra_runner(config_path="config", config_name="nest_fast-conformer")
def main(cfg):
    """Main training function using SimplifiedSSLModel."""
    
    if is_global_rank_zero():
        logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
        logger.info("Using SimplifiedSSLModel for training")
    
    # Create trainer
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Create simplified model
    model = SimplifiedSSLModel(cfg=cfg.model, trainer=trainer)
    
    # Initialize weights from pretrained checkpoint if provided
    model.maybe_init_from_pretrained_checkpoint(cfg)
    
    if is_global_rank_zero():
        logger.info(f"Model created: {type(model).__name__}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer.fit(model)


if __name__ == "__main__":
    main()

