#!/usr/bin/env python3
"""
Inspect source code of stft methods.
"""

import inspect
import sys
from pathlib import Path

# Try to import NeMo
try:
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
    HAVE_NEMO = True
except ImportError:
    HAVE_NEMO = False
    print("NeMo not available.")

# Try to import nest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from nest_ssl_project.modules.audio_preprocessing import FilterbankFeatures as NestFilterbankFeatures
    HAVE_NEST = True
except ImportError:
    HAVE_NEST = False
    print("nest not available.")


def main():
    if HAVE_NEMO:
        print("="*80)
        print("NeMo STFT Source:")
        print("="*80)
        print(inspect.getsource(FilterbankFeatures.stft))
    
    if HAVE_NEST:
        print("="*80)
        print("nest STFT Source:")
        print("="*80)
        print(inspect.getsource(NestFilterbankFeatures.stft))

if __name__ == '__main__':
    main()

