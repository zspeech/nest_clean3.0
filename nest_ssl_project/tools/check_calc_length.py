#!/usr/bin/env python3
"""
Check calc_length results in ConvSubsampling.forward.
"""

import torch
import pickle
from pathlib import Path
import math


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


def main():
    nemo_dir = Path('./saved_nemo_outputs') / 'step_0'
    nest_dir = Path('./saved_nest_outputs') / 'step_0'
    
    print("="*80)
    print("CHECK CALC_LENGTH RESULTS")
    print("="*80)
    
    # Load layer outputs
    with open(nemo_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    # Get preprocessor Call 1 output (seq_len)
    nemo_preproc_outs = nemo_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    nest_preproc_outs = nest_layers.get('preprocessor', {}).get('all_forward_outputs', [])
    
    nemo_seq_len = nemo_preproc_outs[1][1]  # Call 1, output[1] (seq_len)
    nest_seq_len = nest_preproc_outs[1][1]  # Call 1, output[1] (seq_len)
    
    print(f"\nPreprocessor Call 1 seq_len:")
    print(f"  NeMo: {nemo_seq_len}")
    print(f"  nest: {nest_seq_len}")
    
    # Typical ConvSubsampling parameters for striding
    # From config: subsampling='striding', subsampling_factor=4
    # This means: _sampling_num = log2(4) = 2
    # For striding: _stride=2, _kernel_size=3, _left_padding=1, _right_padding=1
    all_paddings = 1 + 1  # _left_padding + _right_padding
    kernel_size = 3
    stride = 2
    ceil_mode = False
    repeat_num = 2  # _sampling_num
    
    print(f"\nConvSubsampling parameters:")
    print(f"  all_paddings: {all_paddings}")
    print(f"  kernel_size: {kernel_size}")
    print(f"  stride: {stride}")
    print(f"  ceil_mode: {ceil_mode}")
    print(f"  repeat_num: {repeat_num}")
    
    # Calculate out_lengths
    nemo_out_lengths = calc_length(nemo_seq_len, all_paddings, kernel_size, stride, ceil_mode, repeat_num)
    nest_out_lengths = calc_length(nest_seq_len, all_paddings, kernel_size, stride, ceil_mode, repeat_num)
    
    print(f"\ncalc_length results:")
    print(f"  NeMo: {nemo_out_lengths}")
    print(f"  nest: {nest_out_lengths}")
    
    if isinstance(nemo_out_lengths, torch.Tensor) and isinstance(nest_out_lengths, torch.Tensor):
        diff = (nemo_out_lengths.float() - nest_out_lengths.float()).abs()
        max_diff = diff.max().item()
        print(f"  Comparison: max_diff={max_diff:.6e} {'[OK]' if max_diff < 1e-5 else '[FAIL]'}")


if __name__ == '__main__':
    main()

