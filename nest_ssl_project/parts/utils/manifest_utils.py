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
Simplified manifest utilities to replace nemo.collections.asr.parts.utils.manifest_utils
"""

import json
from typing import List, Dict, Any, Union
from pathlib import Path


def read_manifest(manifest_filepath: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Read manifest file(s) and return list of entries.
    
    Args:
        manifest_filepath: Path to manifest file or list of paths
    
    Returns:
        List of manifest entries (dictionaries)
    """
    if isinstance(manifest_filepath, str):
        manifest_filepath = [manifest_filepath]
    
    all_entries = []
    for manifest_file in manifest_filepath:
        manifest_file = Path(manifest_file).expanduser().resolve()
        
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    all_entries.append(entry)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error parsing manifest line in {manifest_file}: {line}\nError: {e}")
    
    return all_entries

