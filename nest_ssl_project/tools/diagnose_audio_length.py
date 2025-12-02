"""
Diagnose audio length differences between NeMo and nest implementations.
This script loads the same audio file using both implementations and compares lengths.
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import both implementations
try:
    from nest_ssl_project.parts.preprocessing.segment import AudioSegment as NestAudioSegment
except ImportError:
    # Try alternative import path
    sys.path.insert(0, str(project_root / "nest_ssl_project"))
    from parts.preprocessing.segment import AudioSegment as NestAudioSegment

try:
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment as NeMoAudioSegment
except ImportError:
    print("Warning: Could not import NeMo AudioSegment. Make sure NeMo is in PYTHONPATH.")
    NeMoAudioSegment = None


def diagnose_audio_length(audio_file, offset=0.0, duration=None, target_sr=16000):
    """Compare audio lengths loaded by NeMo and nest."""
    print(f"\n{'='*80}")
    print(f"Diagnosing audio file: {audio_file}")
    print(f"  offset={offset}, duration={duration}, target_sr={target_sr}")
    print(f"{'='*80}")
    
    # Load with nest implementation
    try:
        nest_segment = NestAudioSegment.from_file(
            audio_file,
            offset=offset,
            duration=duration,
            target_sr=target_sr,
        )
        nest_len = nest_segment.samples.shape[0]
        nest_sr = nest_segment.sample_rate
        print(f"\nNest implementation:")
        print(f"  Length: {nest_len} samples")
        print(f"  Sample rate: {nest_sr} Hz")
        print(f"  Duration: {nest_len / nest_sr:.6f} seconds")
    except Exception as e:
        print(f"\nNest implementation ERROR: {e}")
        nest_len = None
        nest_sr = None
    
    # Load with NeMo implementation
    if NeMoAudioSegment is not None:
        try:
            nemo_segment = NeMoAudioSegment.from_file(
                audio_file,
                offset=offset,
                duration=duration if duration is not None else 0,
                target_sr=target_sr,
            )
            nemo_len = nemo_segment.samples.shape[0]
            nemo_sr = nemo_segment.sample_rate
            print(f"\nNeMo implementation:")
            print(f"  Length: {nemo_len} samples")
            print(f"  Sample rate: {nemo_sr} Hz")
            print(f"  Duration: {nemo_len / nemo_sr:.6f} seconds")
        except Exception as e:
            print(f"\nNeMo implementation ERROR: {e}")
            nemo_len = None
            nemo_sr = None
    else:
        nemo_len = None
        nemo_sr = None
    
    # Compare
    if nest_len is not None and nemo_len is not None:
        diff = abs(nemo_len - nest_len)
        print(f"\nComparison:")
        print(f"  Length difference: {diff} samples")
        print(f"  Length difference: {diff / target_sr:.6f} seconds")
        if diff > 0:
            print(f"  [FAIL] Length mismatch!")
            if diff == 160:
                print(f"  [NOTE] Difference is exactly 1 hop_length (160 samples at 16kHz)")
        else:
            print(f"  [OK] Lengths match!")
    
    return nest_len, nemo_len


if __name__ == "__main__":
    # Example: Load from manifest
    # You can modify this to load from your actual manifest
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        offset = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
        duration = float(sys.argv[3]) if len(sys.argv) > 3 else None
        target_sr = int(sys.argv[4]) if len(sys.argv) > 4 else 16000
        
        diagnose_audio_length(audio_file, offset=offset, duration=duration, target_sr=target_sr)
    else:
        print("Usage: python diagnose_audio_length.py <audio_file> [offset] [duration] [target_sr]")
        print("\nExample:")
        print("  python diagnose_audio_length.py /path/to/audio.wav 0.0 12.49 16000")

