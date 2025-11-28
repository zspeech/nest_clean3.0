#!/usr/bin/env python3
"""
Prepare a small dummy SSL dataset for debugging purposes.
Downloads/generates a small batch of audio files and creates manifest files.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import argparse

try:
    import soundfile as sf
    import numpy as np
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False
    print("Warning: soundfile not installed. Install with: pip install soundfile")

try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False
    print("Warning: requests not installed. Install with: pip install requests")


def generate_dummy_audio(output_path: Path, duration: float = 3.0, sample_rate: int = 16000):
    """
    Generate a dummy audio file with random noise.
    
    Args:
        output_path: Path to save the audio file
        duration: Duration in seconds
        sample_rate: Sample rate (default 16000)
    """
    if not HAVE_SOUNDFILE:
        raise ImportError("soundfile is required. Install with: pip install soundfile")
    
    # Generate random audio signal (white noise)
    num_samples = int(duration * sample_rate)
    audio_data = np.random.randn(num_samples).astype(np.float32) * 0.1  # Scale down to avoid clipping
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as WAV file
    sf.write(str(output_path), audio_data, sample_rate)
    print(f"Generated dummy audio: {output_path} ({duration:.2f}s)")


def download_sample_audio(output_path: Path, url: str = None):
    """
    Download a sample audio file from a public URL.
    Falls back to generating dummy audio if download fails.
    
    Args:
        output_path: Path to save the audio file
        url: URL to download from (optional)
    """
    if url and HAVE_REQUESTS:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded audio: {output_path}")
                return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}. Generating dummy audio instead.")
    
    # Fallback to generating dummy audio
    generate_dummy_audio(output_path, duration=3.0)
    return False


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of an audio file in seconds."""
    if not HAVE_SOUNDFILE:
        # Fallback: assume 3 seconds for dummy files
        return 3.0
    
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception as e:
        print(f"Warning: Could not get duration for {audio_path}: {e}. Using default 3.0s")
        return 3.0


def create_manifest_entry(audio_filepath: str, duration: float, text: str = "") -> Dict:
    """
    Create a manifest entry.
    
    Args:
        audio_filepath: Path to audio file (relative to manifest directory)
        duration: Duration in seconds
        text: Text transcription (empty for SSL)
    
    Returns:
        Dictionary representing a manifest entry
    """
    return {
        "audio_filepath": audio_filepath,
        "duration": duration,
        "text": text
    }


def create_manifest(manifest_path: Path, audio_files: List[Path], data_dir: Path):
    """
    Create a manifest file from a list of audio files.
    
    Args:
        manifest_path: Path to save the manifest file
        audio_files: List of audio file paths
        data_dir: Base directory for audio files (for relative paths)
    """
    manifest_entries = []
    
    for audio_file in audio_files:
        # Get relative path from data_dir
        try:
            rel_path = os.path.relpath(audio_file, data_dir)
        except ValueError:
            # If on different drives (Windows), use absolute path
            rel_path = str(audio_file)
        
        duration = get_audio_duration(audio_file)
        
        entry = create_manifest_entry(rel_path, duration)
        manifest_entries.append(entry)
    
    # Write manifest file
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Created manifest: {manifest_path} ({len(manifest_entries)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Prepare dummy SSL dataset for debugging")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to store audio files (default: nest_ssl_project/data/dummy_ssl)"
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=10,
        help="Number of training audio files to generate (default: 10)"
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=2,
        help="Number of validation audio files to generate (default: 2)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration of each audio file in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate for audio files (default: 16000)"
    )
    
    args = parser.parse_args()
    
    # Determine root directory (nest_ssl_project)
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent  # nest_ssl_project
    
    # Set up paths
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = root_dir / "data" / "dummy_ssl"
    
    train_audio_dir = data_dir / "train"
    val_audio_dir = data_dir / "val"
    
    train_manifest = data_dir / "train_manifest.json"
    val_manifest = data_dir / "val_manifest.json"
    
    print(f"Preparing dummy SSL dataset in: {data_dir}")
    print(f"Training files: {args.num_train}, Validation files: {args.num_val}")
    print(f"Duration: {args.duration}s, Sample rate: {args.sample_rate}Hz")
    
    # Check dependencies
    if not HAVE_SOUNDFILE:
        print("\nERROR: soundfile is required. Install with: pip install soundfile")
        sys.exit(1)
    
    # Create directories
    train_audio_dir.mkdir(parents=True, exist_ok=True)
    val_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training audio files
    print("\nGenerating training audio files...")
    train_audio_files = []
    for i in range(args.num_train):
        audio_file = train_audio_dir / f"train_{i:04d}.wav"
        generate_dummy_audio(audio_file, duration=args.duration, sample_rate=args.sample_rate)
        train_audio_files.append(audio_file)
    
    # Generate validation audio files
    print("\nGenerating validation audio files...")
    val_audio_files = []
    for i in range(args.num_val):
        audio_file = val_audio_dir / f"val_{i:04d}.wav"
        generate_dummy_audio(audio_file, duration=args.duration, sample_rate=args.sample_rate)
        val_audio_files.append(audio_file)
    
    # Create manifest files
    print("\nCreating manifest files...")
    create_manifest(train_manifest, train_audio_files, data_dir)
    create_manifest(val_manifest, val_audio_files, data_dir)
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"\nTraining manifest: {train_manifest}")
    print(f"Validation manifest: {val_manifest}")
    print(f"\nTo use this dataset, update your config file:")
    print(f"  model.train_ds.manifest_filepath={train_manifest}")
    print(f"  model.validation_ds.manifest_filepath={val_manifest}")
    print(f"\nOr run training with:")
    print(f"  python train.py \\")
    print(f"    model.train_ds.manifest_filepath={train_manifest} \\")
    print(f"    model.validation_ds.manifest_filepath={val_manifest} \\")
    print(f"    model.train_ds.batch_size=2 \\")
    print(f"    trainer.devices=1")


if __name__ == "__main__":
    main()

