#!/usr/bin/env python3
"""
Capture normalize_batch inputs during actual training run.
This script modifies normalize_batch to save its inputs for debugging.
"""

import torch
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import normalize_batch
from modules.audio_preprocessing import normalize_batch as original_normalize_batch

# Global storage for captured inputs
captured_inputs = {}

def normalize_batch_with_capture(x, seq_len, normalize_type):
    """Wrapper around normalize_batch that captures inputs."""
    # Save inputs
    key = f"call_{len(captured_inputs)}"
    captured_inputs[key] = {
        'x': x.detach().cpu().clone(),
        'seq_len': seq_len.detach().cpu().clone(),
        'normalize_type': normalize_type,
        'x_shape': x.shape,
        'x_dtype': str(x.dtype),
        'x_device': str(x.device),
        'seq_len_shape': seq_len.shape,
        'seq_len_dtype': str(seq_len.dtype),
        'seq_len_values': seq_len.detach().cpu().clone().tolist(),
    }
    
    # Call original
    result = original_normalize_batch(x, seq_len, normalize_type)
    
    # Save outputs
    captured_inputs[key]['output'] = result[0].detach().cpu().clone()
    captured_inputs[key]['mean'] = result[1].detach().cpu().clone() if result[1] is not None else None
    captured_inputs[key]['std'] = result[2].detach().cpu().clone() if result[2] is not None else None
    
    return result

# Replace normalize_batch in the module
import modules.audio_preprocessing as audio_preprocessing_module
audio_preprocessing_module.normalize_batch = normalize_batch_with_capture

print("normalize_batch wrapper installed. Inputs will be captured during training.")
print("After training, check captured_inputs dictionary or save it to a file.")

# Example: Save after first call
def save_captured_inputs(output_path):
    """Save captured inputs to file."""
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(captured_inputs, f)
    print(f"Saved {len(captured_inputs)} captured calls to {output_path}")

if __name__ == '__main__':
    # This is a utility script - import it in training script
    print("Import this module in your training script to capture normalize_batch inputs.")
    print("Example:")
    print("  from tools.capture_normalize_inputs import captured_inputs, save_captured_inputs")
    print("  # ... run training ...")
    print("  save_captured_inputs('normalize_inputs.pkl')")

