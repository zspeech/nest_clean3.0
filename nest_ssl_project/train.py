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


import lightning.pytorch as pl
from omegaconf import OmegaConf

# Import local utilities
from utils.hydra_runner import hydra_runner
from utils.logging import get_logger
from utils.exp_manager import exp_manager

# Import local model
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ssl_models import EncDecDenoiseMaskedTokenPredModel


"""
# Example of training a self-supervised denoising masked token prediction model
```sh
python train.py \
    # (Optional: --config-path=config --config-name=nest_fast-conformer) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.train_ds.noise_manifest=<path to noise manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.validation_ds.noise_manifest=<path to noise manifest> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    strategy="ddp"  \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```
"""


logger = get_logger(__name__)


@hydra_runner(config_path="config", config_name="nest_fast-conformer")
def main(cfg):
    logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()

