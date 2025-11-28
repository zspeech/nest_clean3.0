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
Simplified collections to replace nemo.collections.common.parts.preprocessing.collections
"""

import collections as _collections
from typing import List, Optional, Union, Callable, Dict, Any
from dataclasses import dataclass

from common.parts.preprocessing.manifest import item_iter
from common.parts.preprocessing.parsers import CharParser


class _Collection(_collections.UserList):
    """Base collection class."""
    OUTPUT_TYPE = None
    
    def __init__(self, data):
        super().__init__(data)


@dataclass
class AudioTextItem:
    """Audio text item."""
    id: int
    audio_file: str
    duration: float
    text: str
    offset: Optional[float] = None
    speaker: Optional[int] = None
    orig_sr: Optional[int] = None
    token_labels: Optional[List[int]] = None
    lang: Optional[str] = None
    text_tokens: Optional[List[int]] = None


class AudioText(_Collection):
    """Audio text collection."""
    OUTPUT_TYPE = AudioTextItem
    
    def __init__(
        self,
        ids: List[int],
        audio_files: List[str],
        durations: List[float],
        texts: List[str],
        offsets: List[Optional[float]],
        speakers: List[Optional[int]],
        orig_srs: List[Optional[int]],
        token_labels: List[Optional[List[int]]],
        langs: List[Optional[str]],
        parser: CharParser,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        # Filter by duration
        filtered_data = []
        for i, (audio_file, duration, text, offset, speaker, orig_sr, token_label, lang) in enumerate(
            zip(audio_files, durations, texts, offsets, speakers, orig_srs, token_labels, langs)
        ):
            if min_duration is not None and duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                continue
            
            # Parse text
            text_tokens = parser(text) if text else []
            
            item = AudioTextItem(
                id=ids[i],
                audio_file=audio_file,
                duration=duration,
                text=text,
                offset=offset,
                speaker=speaker,
                orig_sr=orig_sr,
                token_labels=token_label,
                lang=lang,
                text_tokens=text_tokens,
            )
            filtered_data.append(item)
        
        # Limit number
        if max_number is not None and max_number > 0:
            filtered_data = filtered_data[:max_number]
        
        # Sort by duration if requested
        if do_sort_by_duration:
            filtered_data.sort(key=lambda x: x.duration)
        
        super().__init__(filtered_data)
        
        # Create mapping if needed
        self.mapping = {}
        if index_by_file_id:
            for idx, item in enumerate(filtered_data):
                file_id = item.audio_file.split('/')[-1].split('.')[0]
                if file_id not in self.mapping:
                    self.mapping[file_id] = []
                self.mapping[file_id].append(idx)


class ASRAudioText(AudioText):
    """ASR audio text collection from manifest files."""
    
    def __init__(
        self,
        manifests_files: Union[str, List[str]],
        parser: CharParser,
        parse_func: Optional[Callable] = None,
        *args,
        **kwargs
    ):
        ids, audio_files, durations, texts, offsets = [], [], [], [], []
        speakers, orig_srs, token_labels, langs = [], [], [], []
        
        for item in item_iter(manifests_files, parse_func=parse_func):
            ids.append(item.get('id', len(ids)))
            audio_files.append(item['audio_file'])
            durations.append(item.get('duration', 0.0))
            texts.append(item.get('text', ''))
            offsets.append(item.get('offset', None))
            speakers.append(item.get('speaker', None))
            orig_srs.append(item.get('orig_sr', None))
            token_labels.append(item.get('token_labels', None))
            langs.append(item.get('lang', None))
        
        super().__init__(
            ids, audio_files, durations, texts, offsets,
            speakers, orig_srs, token_labels, langs,
            parser, *args, **kwargs
        )

