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
Simplified audio to text dataset utilities to replace nemo.collections.asr.data.audio_to_text_dataset
"""

import copy
from typing import Any, Dict, Optional
from omegaconf import DictConfig, open_dict

from utils.logging import get_logger
from data import audio_to_text

logger = get_logger(__name__)


def inject_dataloader_value_from_model_config(model_cfg: dict, dataloader_cfg: DictConfig, key: str):
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


def get_char_dataset(config: dict, augmentor: Optional[Any] = None) -> audio_to_text.AudioToCharDataset:
    """
    Instantiates a Character Encoding based AudioToCharDataset.
    
    Args:
        config: Config of the AudioToCharDataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.
    
    Returns:
        An instance of AudioToCharDataset.
    """
    if 'labels' not in config:
        logger.warning("dataset does not have explicitly defined labels")
    
    dataset = audio_to_text.AudioToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config.get('labels', None),
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
    )
    return dataset


def get_tarred_dataset(
    config: dict,
    shuffle_n: int,
    global_rank: int,
    world_size: int,
    augmentor: Optional[Any] = None,
) -> audio_to_text.TarredAudioToCharDataset:
    """
    Instantiates a Character Encoding based TarredAudioToCharDataset.
    
    Args:
        config: Config of the TarredAudioToCharDataset.
        shuffle_n: How many samples to look ahead and load to be shuffled.
        global_rank: Global rank of this device.
        world_size: Global world size in the training method.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.
    
    Returns:
        An instance of TarredAudioToCharDataset or a ChainDataset of multiple datasets.
    """
    tarred_audio_filepaths = config['tarred_audio_filepaths']
    manifest_filepaths = config['manifest_filepath']
    
    # Convert to lists if needed
    if isinstance(tarred_audio_filepaths, str):
        tarred_audio_filepaths = [tarred_audio_filepaths]
    if isinstance(manifest_filepaths, str):
        manifest_filepaths = [manifest_filepaths]
    
    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths "
            f"(length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )
    
    if 'labels' not in config:
        logger.warning("dataset does not have explicitly defined labels")
    
    if 'max_utts' in config:
        logger.warning('"max_utts" parameter is not supported for tarred datasets')
    
    datasets = []
    for tarred_audio_filepath, manifest_filepath in zip(tarred_audio_filepaths, manifest_filepaths):
        # Handle single-item lists
        if isinstance(tarred_audio_filepath, list) and len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]
        if isinstance(manifest_filepath, list) and len(manifest_filepath) == 1:
            manifest_filepath = manifest_filepath[0]
        
        dataset = audio_to_text.TarredAudioToCharDataset(
            audio_tar_filepaths=tarred_audio_filepath,
            manifest_filepath=manifest_filepath,
            labels=config.get('labels', None),
            sample_rate=config['sample_rate'],
            int_values=config.get('int_values', False),
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            blank_index=config.get('blank_index', -1),
            unk_index=config.get('unk_index', -1),
            normalize=config.get('normalize_transcripts', False),
            trim=config.get('trim_silence', False),
            parser=config.get('parser', 'en'),
            shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
            shard_manifests=config.get('shard_manifests', False),
            global_rank=global_rank,
            world_size=world_size,
            return_sample_id=config.get('return_sample_id', False),
        )
        datasets.append(dataset)
    
    if len(datasets) == 1:
        return datasets[0]
    else:
        return get_chain_dataset(datasets=datasets, ds_config=config, rank=global_rank)


def get_chain_dataset(datasets, ds_config, rank):
    """
    Create a chain dataset from multiple datasets.
    Aligned with NeMo original implementation to support bucketing strategy.
    """
    import random
    from omegaconf import ListConfig
    
    if len(datasets) > 1:
        if ds_config.get('bucketing_batch_size', None) is not None:
            bucketing_batch_sizes = calc_bucketing_batch_sizes(ds_config, len(datasets))
            logger.info(
                f"Batch bucketing is enabled for {len(datasets)} buckets with adaptive batch sizes of {bucketing_batch_sizes}!"
            )
            for idx, dataset in enumerate(datasets):
                datasets[idx] = audio_to_text.BucketingDataset(
                    dataset=dataset, bucketing_batch_size=bucketing_batch_sizes[idx]
                )
        else:
            logger.info(
                f"Batch bucketing is enabled for {len(datasets)} buckets with fixed batch size of {ds_config['batch_size']}!"
            )

    if len(datasets) == 1:
        return datasets[0]
    
    bucketing_strategy = ds_config.get('bucketing_strategy', 'synced_randomized')
    if bucketing_strategy == 'fixed_order':
        from torch.utils.data import ChainDataset
        return ChainDataset(datasets)
    elif bucketing_strategy == 'synced_randomized':
        return audio_to_text.RandomizedChainDataset(datasets=datasets, rnd_seed=0)
    elif bucketing_strategy == 'fully_randomized':
        return audio_to_text.RandomizedChainDataset(datasets=datasets, rnd_seed=random.randint(0, 30000) + rank)
    else:
        raise ValueError(
            f'bucketing_strategy={bucketing_strategy} is not supported! Supported strategies are [fixed_order, fully_randomized, synced_randomized].'
        )


def calc_bucketing_batch_sizes(ds_config, datasets_len):
    """Calculate bucketing batch sizes from config."""
    from omegaconf import ListConfig
    
    bucketing_batch_size = ds_config['bucketing_batch_size']
    bucketing_weights = ds_config.get('bucketing_weights', None)  # To adjust for upsampled buckets

    bucketing_batch_sizes = []

    if ds_config['batch_size'] != 1:
        raise ValueError(
            f"batch_size should be set to one when bucketing_batch_size is set and adaptive bucketing is enabled (batch_size={ds_config['batch_size']}!"
        )
    if type(bucketing_batch_size) == int:  # linear scaling
        if bucketing_weights:  # Want same batchsize for the same duplicated bucket
            for idx, weight in enumerate(bucketing_weights):
                scale_factor = datasets_len - idx
                [bucketing_batch_sizes.append(scale_factor * bucketing_batch_size) for _ in range(weight)]
        else:
            for idx in range(datasets_len):
                scale_factor = datasets_len - idx
                bucketing_batch_sizes.append(scale_factor * bucketing_batch_size)
    elif isinstance(bucketing_batch_size, ListConfig) or isinstance(
        bucketing_batch_size, list
    ):  # assigned bucket sizes
        if bucketing_weights:  # Want same batchsize for same duplicated bucket
            for idx, weight in enumerate(bucketing_weights):
                [bucketing_batch_sizes.append(bucketing_batch_size[idx]) for _ in range(weight)]
        else:
            bucketing_batch_sizes = bucketing_batch_size
    else:
        raise ValueError(
            f"bucketing_batch_size should be an integer or a list (bucketing_batch_size={bucketing_batch_size})!"
        )

    if len(bucketing_batch_sizes) != datasets_len:
        raise ValueError(
            f"batch_size should have the same length as the number of buckets ({len(bucketing_batch_sizes)}!={datasets_len}) "
        )
    return bucketing_batch_sizes

