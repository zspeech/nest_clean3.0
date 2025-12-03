#!/usr/bin/env python3
"""
Inspect NeMo's ConvSubsampling.forward implementation
"""

import inspect
import sys
sys.path.insert(0, '.')


def main():
    print("="*80)
    print("INSPECT NEMO CONVSUBSAMPLING.FORWARD")
    print("="*80)
    
    from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
    
    # Get source code
    source = inspect.getsource(ConvSubsampling.forward)
    
    print("\nNeMo ConvSubsampling.forward source code:")
    print("-"*60)
    print(source)
    print("-"*60)


if __name__ == '__main__':
    main()

