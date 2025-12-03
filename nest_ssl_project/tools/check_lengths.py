#!/usr/bin/env python3
"""
Check MaskedConvSequential input lengths.
"""

import torch
import pickle
from pathlib import Path

def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK LENGTHS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Check encoder.pre_encode.conv inputs (should be x, lengths)
    # But hooks only capture x usually if registered on forward(x, lengths)?
    # Wait, TrainingOutputSaver.forward_hook captures (args, kwargs) or output?
    # It captures input which is a tuple (x, lengths) if forward takes multiple args.
    
    nemo_conv = nemo_layers.get('encoder.pre_encode.conv', {})
    nest_conv = nest_layers.get('encoder.pre_encode.conv', {})
    
    nemo_inputs = nemo_conv.get('all_forward_inputs', [])
    nest_inputs = nest_conv.get('all_forward_inputs', [])
    
    if len(nemo_inputs) > 0:
        inp = nemo_inputs[0] # (x, lengths)
        if isinstance(inp, (list, tuple)) and len(inp) >= 2:
            lengths = inp[1]
            if isinstance(lengths, torch.Tensor):
                print(f"NeMo lengths: {lengths}")
            else:
                 print(f"NeMo lengths type: {type(lengths)}")
        else:
            print(f"NeMo input structure: {type(inp)}")

    if len(nest_inputs) > 0:
        inp = nest_inputs[0] # (x, lengths)
        if isinstance(inp, (list, tuple)) and len(inp) >= 2:
            lengths = inp[1]
            if isinstance(lengths, torch.Tensor):
                print(f"nest lengths: {lengths}")
            else:
                 print(f"nest lengths type: {type(lengths)}")
        else:
            print(f"nest input structure: {type(inp)}")

if __name__ == '__main__':
    main()

