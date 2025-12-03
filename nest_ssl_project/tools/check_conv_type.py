#!/usr/bin/env python3
"""
Check the actual type of self.conv in ConvSubsampling.
"""

import torch
import pickle
from pathlib import Path
import sys

# Add NeMo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'NeMo'))

from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel
from omegaconf import OmegaConf


def main():
    # Load NeMo config
    config_path = Path(__file__).parent.parent.parent / 'NeMo' / 'examples' / 'asr' / 'conf' / 'ssl' / 'nest' / 'nest_fast-conformer.yaml'
    cfg = OmegaConf.load(config_path)
    
    # Create model
    print("Creating NeMo model...")
    model = EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=None)
    
    # Check encoder.pre_encode type
    print(f"\nencoder.pre_encode type: {type(model.encoder.pre_encode)}")
    print(f"encoder.pre_encode.__class__.__name__: {model.encoder.pre_encode.__class__.__name__}")
    
    # Check encoder.pre_encode.conv type
    if hasattr(model.encoder.pre_encode, 'conv'):
        print(f"\nencoder.pre_encode.conv type: {type(model.encoder.pre_encode.conv)}")
        print(f"encoder.pre_encode.conv.__class__.__name__: {model.encoder.pre_encode.conv.__class__.__name__}")
        print(f"encoder.pre_encode.conv.__class__.__module__: {model.encoder.pre_encode.conv.__class__.__module__}")
        
        # Check if it's Sequential or MaskedConvSequential
        import torch.nn as nn
        if isinstance(model.encoder.pre_encode.conv, nn.Sequential):
            print("\n✓ encoder.pre_encode.conv is nn.Sequential")
        else:
            print(f"\n✗ encoder.pre_encode.conv is NOT nn.Sequential")
        
        # Check for MaskedConvSequential
        try:
            from nemo.collections.asr.parts.submodules.subsampling import MaskedConvSequential
            if isinstance(model.encoder.pre_encode.conv, MaskedConvSequential):
                print("✓ encoder.pre_encode.conv is MaskedConvSequential")
            else:
                print("✗ encoder.pre_encode.conv is NOT MaskedConvSequential")
        except ImportError:
            print("✗ MaskedConvSequential not available in this NeMo version")
    else:
        print("\nencoder.pre_encode.conv attribute not found")
    
    # Check conv2d_subsampling flag
    if hasattr(model.encoder.pre_encode, 'conv2d_subsampling'):
        print(f"\nencoder.pre_encode.conv2d_subsampling: {model.encoder.pre_encode.conv2d_subsampling}")
    
    # Check subsampling_conv_chunking_factor
    if hasattr(model.encoder.pre_encode, 'subsampling_conv_chunking_factor'):
        print(f"encoder.pre_encode.subsampling_conv_chunking_factor: {model.encoder.pre_encode.subsampling_conv_chunking_factor}")
    
    # Check subsampling type
    if hasattr(model.encoder.pre_encode, '_subsampling'):
        print(f"encoder.pre_encode._subsampling: {model.encoder.pre_encode._subsampling}")


if __name__ == '__main__':
    main()

