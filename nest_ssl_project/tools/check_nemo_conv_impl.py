#!/usr/bin/env python3
"""
Check NeMo's ConvSubsampling implementation
"""

import sys
sys.path.insert(0, '.')

import inspect


def main():
    print("="*80)
    print("CHECK NEMO CONVSUBSAMPLING IMPLEMENTATION")
    print("="*80)
    
    try:
        from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
        
        # Get the source file
        source_file = inspect.getfile(ConvSubsampling)
        print(f"\nNeMo ConvSubsampling source file: {source_file}")
        
        # Get the source code
        source = inspect.getsource(ConvSubsampling)
        
        # Check if MaskedConvSequential is used
        if 'MaskedConvSequential' in source:
            print("\n[INFO] NeMo uses MaskedConvSequential")
        else:
            print("\n[WARNING] NeMo does NOT use MaskedConvSequential!")
            print("This means NeMo uses nn.Sequential without masking.")
        
        # Check what self.conv is assigned to
        print("\n--- Checking self.conv assignment ---")
        for line in source.split('\n'):
            if 'self.conv =' in line or 'self.conv=' in line:
                print(f"  {line.strip()}")
        
        # Check the forward method
        print("\n--- Forward method signature ---")
        forward_source = inspect.getsource(ConvSubsampling.forward)
        # Print first 30 lines
        for i, line in enumerate(forward_source.split('\n')[:30]):
            print(f"  {line}")
        
        # Try to check if MaskedConvSequential exists
        print("\n--- Checking MaskedConvSequential availability ---")
        try:
            from nemo.collections.asr.parts.submodules.subsampling import MaskedConvSequential
            print("  MaskedConvSequential is available")
        except ImportError:
            print("  MaskedConvSequential is NOT available in this NeMo version")
        
        # Check if apply_channel_mask exists
        print("\n--- Checking apply_channel_mask availability ---")
        try:
            from nemo.collections.asr.parts.submodules.subsampling import apply_channel_mask
            print("  apply_channel_mask is available")
        except ImportError:
            print("  apply_channel_mask is NOT available in this NeMo version")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

