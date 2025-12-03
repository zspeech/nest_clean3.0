#!/usr/bin/env python3
"""
Check quantizer outputs.
"""

import torch
import pickle
from pathlib import Path

def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK QUANTIZER")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Check quantizer
    nemo_quant = nemo_layers.get('quantizer', {})
    nest_quant = nest_layers.get('quantizer', {})
    
    nemo_out = nemo_quant.get('all_forward_outputs', [])
    nest_out = nest_quant.get('all_forward_outputs', [])
    
    if len(nemo_out) > 0 and len(nest_out) > 0:
        # quantizer output is usually (loss, tokens) or just tokens depending on implementation
        # In NeMo RandomProjectionVectorQuantizer: return loss, probs (or tokens)
        
        nemo_res = nemo_out[0]
        nest_res = nest_out[0]
        
        if isinstance(nemo_res, (list, tuple)):
            print(f"NeMo quantizer output tuple len: {len(nemo_res)}")
            for i, item in enumerate(nemo_res):
                if isinstance(item, torch.Tensor):
                    print(f"  Item {i}: shape={item.shape}, mean={item.float().mean():.6f}, min={item.min()}, max={item.max()}")
        
        if isinstance(nest_res, (list, tuple)):
            print(f"nest quantizer output tuple len: {len(nest_res)}")
            for i, item in enumerate(nest_res):
                if isinstance(item, torch.Tensor):
                    print(f"  Item {i}: shape={item.shape}, mean={item.float().mean():.6f}, min={item.min()}, max={item.max()}")

        # Compare
        if isinstance(nemo_res, (list, tuple)) and isinstance(nest_res, (list, tuple)):
             for i in range(min(len(nemo_res), len(nest_res))):
                 if isinstance(nemo_res[i], torch.Tensor) and isinstance(nest_res[i], torch.Tensor):
                     diff = (nemo_res[i].float() - nest_res[i].float()).abs().max().item()
                     print(f"Item {i} max diff: {diff:.6e} {'[OK]' if diff < 1e-5 else '[FAIL]'}")

if __name__ == '__main__':
    main()

