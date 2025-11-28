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
Manifest processing utilities - extended version
"""

import json
import os
from typing import Iterator, List, Union, Optional, Dict, Any, Callable
from pathlib import Path


def _find_project_root() -> Optional[Path]:
    """Find the nest_ssl_project root directory by looking for train.py."""
    current = Path.cwd()
    
    # Check current directory
    if (current / "train.py").exists():
        return current
    
    # Check nest_ssl_project subdirectory
    if (current / "nest_ssl_project" / "train.py").exists():
        return current / "nest_ssl_project"
    
    # Check parent directories
    for parent in current.parents:
        if (parent / "nest_ssl_project" / "train.py").exists():
            return parent / "nest_ssl_project"
        elif (parent / "train.py").exists():
            return parent
    
    return None


def item_iter(
    manifests_files: Union[str, List[str]],
    parse_func: Optional[Callable[[str, Optional[str]], Dict[str, Any]]] = None
) -> Iterator[Dict[str, Any]]:
    """
    Iterate over manifest items.
    
    Args:
        manifests_files: Path(s) to manifest file(s)
        parse_func: Optional function to parse each line
    
    Yields:
        Dictionary items from manifest
    """
    if isinstance(manifests_files, str):
        manifests_files = [manifests_files]
    
    for manifest_file in manifests_files:
        # Convert to Path and expand user home directory
        manifest_path = Path(manifest_file).expanduser()
        
        # If relative path, try multiple locations
        if not manifest_path.is_absolute():
            tried_paths = []
            resolved_path = None
            
            # Try 1: Current working directory
            cwd_path = Path.cwd() / manifest_path
            tried_paths.append(str(cwd_path))
            if cwd_path.exists():
                resolved_path = cwd_path.resolve()
            else:
                # Try 2: Relative to project root (nest_ssl_project)
                project_root = _find_project_root()
                if project_root:
                    project_path = project_root / manifest_path
                    tried_paths.append(str(project_path))
                    if project_path.exists():
                        resolved_path = project_path.resolve()
                
                # Try 3: Relative to current directory (last resort)
                if resolved_path is None:
                    resolved_path = manifest_path.resolve()
                    tried_paths.append(str(resolved_path))
            
            manifest_file = str(resolved_path) if resolved_path else str(manifest_path.resolve())
        else:
            manifest_file = str(manifest_path.resolve())
        
        manifest_file = Path(manifest_file)
        
        if not manifest_file.exists():
            # Provide helpful error message
            cwd = Path.cwd()
            project_root = _find_project_root()
            
            tried_paths = [str(cwd / manifest_path)]
            if project_root:
                tried_paths.append(str(project_root / manifest_path))
            
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_file}\n"
                f"Original path: {manifest_file}\n"
                f"Current working directory: {cwd}\n"
                f"Project root: {project_root}\n"
                f"Tried paths:\n" + "\n".join(f"  - {p}" for p in tried_paths) + "\n"
                f"Please ensure the manifest file exists or provide an absolute path."
            )
        
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if parse_func:
                    item = parse_func(line, str(manifest_file))
                else:
                    item = __parse_item(line, str(manifest_file))
                
                if item:
                    yield item


def __parse_item(line: str, manifest_file: str) -> Dict[str, Any]:
    """Parse a single manifest line."""
    item = json.loads(line)
    
    # Audio file
    if 'audio_filename' in item:
        item['audio_file'] = item.pop('audio_filename')
    elif 'audio_filepath' in item:
        item['audio_file'] = item.pop('audio_filepath')
    else:
        raise KeyError(f"No 'audio_filename' or 'audio_filepath' in manifest item: {item}")
    
    # Resolve audio file path
    audio_file = item['audio_file']
    if not os.path.isabs(audio_file):
        data_dir = os.path.dirname(manifest_file)
        audio_file_path = os.path.join(data_dir, audio_file)
        if os.path.isfile(audio_file_path):
            audio_file = os.path.abspath(audio_file_path)
        else:
            audio_file = os.path.abspath(os.path.expanduser(audio_file))
    item['audio_file'] = audio_file
    
    # Duration
    if 'duration' not in item:
        item['duration'] = None
    
    # Text
    if 'text' not in item:
        item['text'] = ""
    
    # Other fields
    item = dict(
        id=item.get('id', len(item)),
        audio_file=item['audio_file'],
        duration=item['duration'],
        text=item['text'],
        offset=item.get('offset', None),
        speaker=item.get('speaker', None),
        orig_sr=item.get('orig_sample_rate', None),
        token_labels=item.get('token_labels', None),
        lang=item.get('lang', None),
    )
    
    return item


def get_full_path(
    audio_file: Union[str, List[str]],
    manifest_file: Optional[str] = None,
    data_dir: Optional[str] = None,
    audio_file_len_limit: int = 255,
) -> Union[str, List[str]]:
    """
    Get full path to audio_file.
    
    If the audio_file is a relative path and does not exist,
    try to attach the parent directory of manifest to the audio path.
    Revert to the original path if the new path still doesn't exist.
    Assume that the audio path is like "wavs/xxxxxx.wav".
    
    Args:
        audio_file: path to an audio file, either absolute or assumed relative
                    to the manifest directory or data directory.
                    Alternatively, a list of paths may be provided.
        manifest_file: path to a manifest file
        data_dir: path to a directory containing data, use only if a manifest file is not provided
        audio_file_len_limit: limit for length of audio_file when using relative paths
    
    Returns:
        Full path to audio_file or a list of paths.
    """
    if isinstance(audio_file, list):
        return [get_full_path(audio_file=af, manifest_file=manifest_file, data_dir=data_dir, audio_file_len_limit=audio_file_len_limit) for af in audio_file]
    
    if isinstance(audio_file, str):
        # If already absolute and exists, return as is
        if os.path.isabs(audio_file) and os.path.isfile(audio_file):
            return os.path.abspath(audio_file)
        
        # If path is too long, assume it's already a full path
        if len(audio_file) >= audio_file_len_limit:
            return os.path.abspath(os.path.expanduser(audio_file))
        
        # If not absolute and path is short, try to resolve relative to manifest or data_dir
        if not os.path.isabs(audio_file):
            if manifest_file is None and data_dir is None:
                raise ValueError('Use either manifest_file or data_dir to specify the data directory.')
            elif manifest_file is not None and data_dir is not None:
                raise ValueError(
                    f'Parameters manifest_file and data_dir cannot be used simultaneously. '
                    f'Currently manifest_file is {manifest_file} and data_dir is {data_dir}.'
                )
            
            # Resolve the data directory
            if data_dir is None:
                data_dir = os.path.dirname(manifest_file)
            
            # Assume audio_file path is relative to data_dir
            audio_file_path = os.path.join(data_dir, audio_file)
            
            if os.path.isfile(audio_file_path):
                audio_file = os.path.abspath(audio_file_path)
            else:
                # Try expanding user home directory
                audio_file = os.path.abspath(os.path.expanduser(audio_file))
        else:
            audio_file = os.path.abspath(os.path.expanduser(audio_file))
        
        return audio_file
    else:
        raise ValueError(f'Unexpected audio_file type {type(audio_file)}, audio_file {audio_file}.')
