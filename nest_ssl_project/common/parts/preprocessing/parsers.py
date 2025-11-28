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
Simplified text parsers to replace nemo.collections.common.parts.preprocessing.parsers
"""

import string
from typing import List, Optional


class CharParser:
    """Functor for parsing raw strings into list of int tokens."""
    
    def __init__(
        self,
        labels: List[str],
        *,
        unk_id: int = -1,
        blank_id: int = -1,
        do_normalize: bool = True,
        do_lowercase: bool = True,
        do_tokenize: bool = True,
    ):
        """Creates simple mapping char parser."""
        self._labels = labels
        self._unk_id = unk_id
        self._blank_id = blank_id
        self._do_normalize = do_normalize
        self._do_lowercase = do_lowercase
        self._do_tokenize = do_tokenize
        
        self._labels_map = {label: index for index, label in enumerate(labels)}
        self._special_labels = set([label for label in labels if len(label) > 1])
    
    def __call__(self, text: str) -> Optional[List[int]]:
        """Parse text into token indices."""
        if self._do_normalize:
            text = self._normalize(text)
            if text is None:
                return None
        
        if not self._do_tokenize:
            return text
        
        text_tokens = self._tokenize(text)
        return text_tokens
    
    def _normalize(self, text: str) -> Optional[str]:
        """Normalize text."""
        text = text.strip()
        if self._do_lowercase:
            text = text.lower()
        return text
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text into indices."""
        tokens = []
        i = 0
        while i < len(text):
            # Try to match special labels first
            matched = False
            for special_label in self._special_labels:
                if text[i:].startswith(special_label):
                    tokens.append(self._labels_map[special_label])
                    i += len(special_label)
                    matched = True
                    break
            
            if not matched:
                char = text[i]
                if char in self._labels_map:
                    tokens.append(self._labels_map[char])
                elif self._unk_id >= 0:
                    tokens.append(self._unk_id)
                i += 1
        
        # Filter out blank_id
        if self._blank_id >= 0:
            tokens = [t for t in tokens if t != self._blank_id]
        
        return tokens


class ENCharParser(CharParser):
    """English-specific parser."""
    
    PUNCTUATION_TO_REPLACE = {'+': 'plus', '&': 'and', '%': 'percent'}
    
    def __init__(self, abbreviation_version=None, make_table=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._table = None
        if make_table:
            self._table = self.__make_trans_table()
        self.abbreviation_version = abbreviation_version
    
    def __make_trans_table(self):
        """Create translation table for punctuation."""
        punctuation = string.punctuation
        for char in self.PUNCTUATION_TO_REPLACE:
            punctuation = punctuation.replace(char, '')
        for label in self._labels:
            punctuation = punctuation.replace(label, '')
        return str.maketrans(punctuation, ' ' * len(punctuation))
    
    def _normalize(self, text: str) -> Optional[str]:
        """Normalize English text."""
        try:
            # Simple normalization - replace punctuation with spaces
            if self._table:
                text = text.translate(self._table)
            text = text.strip()
            if self._do_lowercase:
                text = text.lower()
            return text
        except Exception:
            return None


NAME_TO_PARSER = {'base': CharParser, 'en': ENCharParser}


def make_parser(
    labels: Optional[List[str]] = None,
    name: str = 'base',
    **kwargs,
) -> CharParser:
    """Creates parser from labels and parser name."""
    if name not in NAME_TO_PARSER:
        raise ValueError(f'Invalid parser name: {name}')
    
    if labels is None:
        labels = list(string.printable)
    
    parser_type = NAME_TO_PARSER[name]
    parser = parser_type(labels=labels, **kwargs)
    
    return parser

