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
Audio to text dataset classes for ASR training.
Simplified version adapted from NeMo's audio_to_text.py
"""

import io
import os
from collections.abc import Iterable as IterableABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import braceexpand
import torch
from torch.utils.data import Dataset, IterableDataset

# Import from local modules
from parts.preprocessing.features import WaveformFeaturizer
from parts.preprocessing.segment import ChannelSelectorType, available_formats as valid_sf_formats

# Import local modules
from common.parts.preprocessing import collections, parsers
from utils.logging import get_logger

# Optional webdataset support
try:
    import webdataset as wds
    HAVE_WEBDATASET = True
    # Simple webdataset_split_by_workers implementation
    def webdataset_split_by_workers(src):
        """Split webdataset by workers (simplified version)."""
        return src
except ImportError:
    HAVE_WEBDATASET = False
    wds = None
    def webdataset_split_by_workers(src):
        return src

logger = get_logger(__name__)

__all__ = [
    'AudioToCharDataset',
    'TarredAudioToCharDataset',
]

VALID_FILE_FORMATS = ';'.join(['wav', 'mp3', 'flac', 'opus'] + [fmt.lower() for fmt in valid_sf_formats.keys()])


def _speech_collate_fn(batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len"""
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 5:
        _, audio_lengths, _, tokens_lengths, sample_ids = packed_batch
    elif len(packed_batch) == 4:
        sample_ids = None
        _, audio_lengths, _, tokens_lengths = packed_batch
    else:
        raise ValueError("Expects 4 or 5 tensors in the batch!")
    
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    has_tokens = tokens_lengths[0] is not None
    if has_tokens:
        max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    for b in batch:
        if len(b) == 5:
            sig, sig_len, tokens_i, tokens_i_len, _ = b
        else:
            sig, sig_len, tokens_i, tokens_i_len = b
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        if has_tokens:
            tokens_i_len = tokens_i_len.item()
            if tokens_i_len < max_tokens_len:
                pad = (0, max_tokens_len - tokens_i_len)
                tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
            tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    if has_tokens:
        tokens = torch.stack(tokens)
        tokens_lengths = torch.stack(tokens_lengths)
    else:
        tokens, tokens_lengths = None, None
    
    if sample_ids is None:
        return audio_signal, audio_lengths, tokens, tokens_lengths
    else:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return audio_signal, audio_lengths, tokens, tokens_lengths, sample_ids


class ASRManifestProcessor:
    """
    Class that processes a manifest json file containing paths to audio files, transcripts, and durations.
    """
    
    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        index_by_file_id: bool = False,
        manifest_parse_func: Optional[Callable] = None,
    ):
        self.parser = parser
        
        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
            parse_func=manifest_parse_func,
        )
        
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id
    
    def process_text_by_id(self, index: int) -> Tuple[List[int], int]:
        sample = self.collection[index]
        return self.process_text_by_sample(sample)
    
    def process_text_by_file_id(self, file_id: str) -> Tuple[List[int], int]:
        manifest_idx = self.collection.mapping[file_id][0]
        sample = self.collection[manifest_idx]
        return self.process_text_by_sample(sample)
    
    def process_text_by_sample(self, sample: collections.ASRAudioText.OUTPUT_TYPE) -> Tuple[List[int], int]:
        t, tl = sample.text_tokens, len(sample.text_tokens)
        
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1
        
        return t, tl


def expand_sharded_filepaths(sharded_filepaths, shard_strategy: str, world_size: int, global_rank: int):
    """Expand and shard filepaths for distributed training."""
    valid_shard_strategies = ['scatter', 'replicate']
    if shard_strategy not in valid_shard_strategies:
        raise ValueError(f"`shard_strategy` must be one of {valid_shard_strategies}")
    
    if isinstance(sharded_filepaths, str):
        # Replace '(' and '[' with '{'
        brace_keys_open = ['(', '[', '<', '_OP_']
        for bkey in brace_keys_open:
            if bkey in sharded_filepaths:
                sharded_filepaths = sharded_filepaths.replace(bkey, "{")
        
        # Replace ')' and ']' with '}'
        brace_keys_close = [')', ']', '>', '_CL_']
        for bkey in brace_keys_close:
            if bkey in sharded_filepaths:
                sharded_filepaths = sharded_filepaths.replace(bkey, "}")
        
        # Brace expand
        sharded_filepaths = list(braceexpand.braceexpand(sharded_filepaths, escape=False))
    
    # Check for distributed and partition shards accordingly
    if world_size > 1:
        if shard_strategy == 'scatter':
            logger.info("All tarred dataset shards will be scattered evenly across all nodes.")
            
            if len(sharded_filepaths) % world_size != 0:
                logger.warning(
                    f"Number of shards in tarred dataset ({len(sharded_filepaths)}) is not divisible "
                    f"by number of distributed workers ({world_size})."
                )
            
            begin_idx = (len(sharded_filepaths) // world_size) * global_rank
            end_idx = begin_idx + len(sharded_filepaths) // world_size
            sharded_filepaths = sharded_filepaths[begin_idx:end_idx]
            logger.info(
                "Partitioning tarred dataset: process (%d) taking shards [%d, %d]", global_rank, begin_idx, end_idx
            )
        elif shard_strategy == 'replicate':
            logger.info("All tarred dataset shards will be replicated across all nodes.")
    
    return sharded_filepaths


class _AudioTextDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations.
    """
    
    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional[Any] = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        manifest_parse_func: Optional[Callable] = None,
    ):
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(",")
        
        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            manifest_parse_func=manifest_parse_func,
        )
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id
        self.channel_selector = channel_selector
    
    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]
    
    def __getitem__(self, index):
        if isinstance(index, IterableABC):
            return [self._process_sample(_index) for _index in index]
        else:
            return self._process_sample(index)
    
    def _process_sample(self, index):
        sample = self.manifest_processor.collection[index]
        offset = sample.offset
        
        if offset is None:
            offset = 0
        
        features = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        f, fl = features, torch.tensor(features.shape[0]).long()
        
        t, tl = self.manifest_processor.process_text_by_sample(sample=sample)
        
        if self.return_sample_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), index
        else:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()
        
        return output
    
    def __len__(self):
        return len(self.manifest_processor.collection)
    
    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.manifest_processor.pad_id)


class AudioToCharDataset(_AudioTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations.
    Uses character-level encoding.
    """
    
    def __init__(
        self,
        manifest_filepath: str,
        labels: Union[str, List[str]],
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional[Any] = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        parser: Union[str, Callable] = 'en',
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        manifest_parse_func: Optional[Callable] = None,
    ):
        self.labels = labels
        
        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )
        
        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=parser,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
            manifest_parse_func=manifest_parse_func,
        )


class _TarredAudioToTextDataset(IterableDataset):
    """
    A similar Dataset to the AudioToCharDataset, but which loads tarred audio files.
    """
    
    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        parser: Callable,
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional[Any] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        shard_strategy: str = "scatter",
        shard_manifests: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
        manifest_parse_func: Optional[Callable] = None,
    ):
        if not HAVE_WEBDATASET:
            raise ImportError("WebDataset is required for TarredAudioToTextDataset. Please install nemo.")
        
        self.shard_manifests = shard_manifests
        
        # Shard manifests if necessary
        if shard_manifests and world_size > 1:
            if isinstance(manifest_filepath, str):
                manifest_filepath = manifest_filepath.split(",")
            # Simple sharding - take every world_size-th manifest starting from global_rank
            if len(manifest_filepath) > 1:
                manifest_filepath = manifest_filepath[global_rank::world_size]
            manifest_filepath = ",".join(manifest_filepath) if isinstance(manifest_filepath, list) else manifest_filepath
        
        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=0,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            index_by_file_id=True,
            manifest_parse_func=manifest_parse_func,
        )
        
        self.len = len(self.manifest_processor.collection)
        
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.return_sample_id = return_sample_id
        
        audio_tar_filepaths = expand_sharded_filepaths(
            sharded_filepaths=audio_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )
        
        # Put together WebDataset pipeline
        self._dataset = wds.DataPipeline(
            wds.SimpleShardList(urls=audio_tar_filepaths),
            webdataset_split_by_workers,
            wds.shuffle(shuffle_n),
            wds.tarfile_to_samples(),
            wds.rename(audio=VALID_FILE_FORMATS, key='__key__'),
            wds.to_tuple('audio', 'key'),
            self._filter,
            self._loop_offsets,
            wds.map(self._build_sample),
        )
    
    def _filter(self, iterator):
        """Filter samples that have been filtered out by ASRAudioText."""
        class TarredAudioFilter:
            def __init__(self, collection):
                self.iterator = iterator
                self.collection = collection
            
            def __iter__(self):
                return self
            
            def __next__(self):
                while True:
                    audio_bytes, audio_filename = next(self.iterator)
                    file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                    if file_id in self.collection.mapping:
                        return audio_bytes, audio_filename
        
        return TarredAudioFilter(self.manifest_processor.collection)
    
    def _loop_offsets(self, iterator):
        """Iterate through utterances with different offsets for each file."""
        class TarredAudioLoopOffsets:
            def __init__(self, collection):
                self.iterator = iterator
                self.collection = collection
                self.current_fn = None
                self.current_bytes = None
                self.offset_id = 0
            
            def __iter__(self):
                return self
            
            def __next__(self):
                if self.current_fn is None:
                    self.current_bytes, self.current_fn = next(self.iterator)
                    self.offset_id = 0
                else:
                    file_id, _ = os.path.splitext(os.path.basename(self.current_fn))
                    offset_list = self.collection.mapping[file_id]
                    if len(offset_list) == self.offset_id + 1:
                        self.current_bytes, self.current_fn = next(self.iterator)
                        self.offset_id = 0
                    else:
                        self.offset_id += 1
                
                return self.current_bytes, self.current_fn, self.offset_id
        
        return TarredAudioLoopOffsets(self.manifest_processor.collection)
    
    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, self.pad_id)
    
    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info."""
        audio_bytes, audio_filename, offset_id = tup
        
        # Grab manifest entry
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))
        manifest_idx = self.manifest_processor.collection.mapping[file_id][offset_id]
        manifest_entry = self.manifest_processor.collection[manifest_idx]
        
        offset = manifest_entry.offset
        if offset is None:
            offset = 0
        
        # Convert audio bytes to IO stream for processing
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream,
            offset=offset,
            duration=manifest_entry.duration,
            trim=self.trim,
            orig_sr=manifest_entry.orig_sr,
        )
        audio_filestream.close()
        
        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()
        
        # Text features
        t, tl = self.manifest_processor.process_text_by_sample(sample=manifest_entry)
        
        if self.return_sample_id:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), manifest_idx
        else:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()
    
    def __iter__(self):
        return self._dataset.__iter__()
    
    def __len__(self):
        return self.len


class TarredAudioToCharDataset(_TarredAudioToTextDataset):
    """
    A similar Dataset to the AudioToCharDataset, but which loads tarred audio files.
    Uses character-level encoding.
    """
    
    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        labels: Union[str, List[str]],
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional[Any] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        parser: Union[str, Callable] = 'en',
        shard_strategy: str = "scatter",
        shard_manifests: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
        manifest_parse_func: Optional[Callable] = None,
    ):
        self.labels = labels
        
        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )
        
        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            parser=parser,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            shard_strategy=shard_strategy,
            shard_manifests=shard_manifests,
            global_rank=global_rank,
            world_size=world_size,
            return_sample_id=return_sample_id,
            manifest_parse_func=manifest_parse_func,
        )

