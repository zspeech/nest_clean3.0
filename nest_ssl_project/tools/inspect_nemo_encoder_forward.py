#!/usr/bin/env python3
"""
Inspect NeMo's ConformerEncoder.forward_internal implementation
"""

import inspect
import sys
sys.path.insert(0, '.')


def main():
    print("="*80)
    print("INSPECT NEMO CONFORMERENCODER.FORWARD_INTERNAL")
    print("="*80)
    
    from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
    
    # Get source code of forward_internal
    try:
        source = inspect.getsource(ConformerEncoder.forward_internal)
        print("\nNeMo ConformerEncoder.forward_internal source code:")
        print("-"*60)
        # Print first 100 lines
        lines = source.split('\n')
        for i, line in enumerate(lines[:100]):
            print(f"{i+1:3d}| {line}")
        if len(lines) > 100:
            print(f"... ({len(lines) - 100} more lines)")
        print("-"*60)
    except AttributeError:
        print("forward_internal not found, trying forward")
        source = inspect.getsource(ConformerEncoder.forward)
        print("\nNeMo ConformerEncoder.forward source code:")
        print("-"*60)
        lines = source.split('\n')
        for i, line in enumerate(lines[:100]):
            print(f"{i+1:3d}| {line}")
        print("-"*60)


if __name__ == '__main__':
    main()

