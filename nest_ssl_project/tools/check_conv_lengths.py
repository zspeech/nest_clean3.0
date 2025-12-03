#!/usr/bin/env python3
"""
Check lengths passed to ConvSubsampling and MaskedConvSequential.
"""

import torch
import pickle
from pathlib import Path


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK CONV LENGTHS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get preprocessor Call 1 output (seq_len)
    nemo_preproc_outs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preproc_outs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    
    if len(nemo_preproc_outs) < 2 or len(nest_preproc_outs) < 2:
        print("ERROR: Not enough preprocessor calls")
        return
    
    nemo_preproc_seq_len = nemo_preproc_outs[1][1]  # Call 1, output[1] (seq_len)
    nest_preproc_seq_len = nest_preproc_outs[1][1]  # Call 1, output[1] (seq_len)
    
    print(f"\nPreprocessor Call 1 seq_len:")
    print(f"  NeMo: {nemo_preproc_seq_len}")
    print(f"  nest: {nest_preproc_seq_len}")
    
    if isinstance(nemo_preproc_seq_len, torch.Tensor) and isinstance(nest_preproc_seq_len, torch.Tensor):
        diff = (nemo_preproc_seq_len.float() - nest_preproc_seq_len.float()).abs()
        max_diff = diff.max().item()
        print(f"  Comparison: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")
    
    # Get pre_encode inputs (lengths)
    nemo_pre_encode = nemo_layers.get('encoder.pre_encode', {})
    nest_pre_encode = nest_layers.get('encoder.pre_encode', {})
    
    nemo_pre_encode_inputs = nemo_pre_encode.get('all_forward_inputs', [])
    nest_pre_encode_inputs = nest_pre_encode.get('all_forward_inputs', [])
    
    print(f"\npre_encode forward_inputs:")
    print(f"  NeMo: {len(nemo_pre_encode_inputs)} calls")
    print(f"  nest: {len(nest_pre_encode_inputs)} calls")
    
    if len(nemo_pre_encode_inputs) > 0 and len(nest_pre_encode_inputs) > 0:
        nemo_pre_encode_input = nemo_pre_encode_inputs[0]
        nest_pre_encode_input = nest_pre_encode_inputs[0]
        
        if isinstance(nemo_pre_encode_input, (list, tuple)) and len(nemo_pre_encode_input) > 1:
            nemo_lengths = nemo_pre_encode_input[1]
            print(f"\nNeMo pre_encode lengths (input[1]): {nemo_lengths}")
        
        if isinstance(nest_pre_encode_input, (list, tuple)) and len(nest_pre_encode_input) > 1:
            nest_lengths = nest_pre_encode_input[1]
            print(f"nest pre_encode lengths (input[1]): {nest_lengths}")
        
        if isinstance(nemo_pre_encode_input, (list, tuple)) and len(nemo_pre_encode_input) > 1 and \
           isinstance(nest_pre_encode_input, (list, tuple)) and len(nest_pre_encode_input) > 1:
            nemo_lengths = nemo_pre_encode_input[1]
            nest_lengths = nest_pre_encode_input[1]
            
            if isinstance(nemo_lengths, torch.Tensor) and isinstance(nest_lengths, torch.Tensor):
                diff = (nemo_lengths.float() - nest_lengths.float()).abs()
                max_diff = diff.max().item()
                print(f"\nComparison: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")


if __name__ == '__main__':
    main()

