#!/usr/bin/env python
"""
Script to synchronize weights from NeMo saved outputs to nest_ssl_project.
This ensures both frameworks use identical weights for comparison.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

import torch


def load_model_structure(pt_file):
    """Load the model structure from a .pt file."""
    data = torch.load(pt_file, map_location='cpu', weights_only=False)
    return data


def sync_weights(nemo_dir, nest_dir, step=None):
    """
    Sync weights from NeMo outputs to nest outputs.
    
    This copies the model_structure.pt file from NeMo to nest,
    effectively making both models start with identical weights.
    """
    nemo_path = Path(nemo_dir)
    nest_path = Path(nest_dir)
    
    # Auto-detect step if not provided
    if step is None:
        # Find common step directories
        nemo_steps = set(d.name for d in nemo_path.iterdir() if d.is_dir() and d.name.startswith('step_'))
        if not nemo_steps:
            print(f"No step directories found in {nemo_path}")
            return False
        step = sorted(nemo_steps, key=lambda x: int(x.split('_')[1]))[0]
        print(f"Auto-detected step: {step}")
    
    nemo_step_dir = nemo_path / step
    nest_step_dir = nest_path / step
    
    if not nemo_step_dir.exists():
        print(f"NeMo step directory does not exist: {nemo_step_dir}")
        return False
    
    # Create nest step directory if it doesn't exist
    nest_step_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model_structure.pt
    nemo_model_file = nemo_step_dir / 'model_structure.pt'
    nest_model_file = nest_step_dir / 'model_structure.pt'
    
    if nemo_model_file.exists():
        print(f"Copying model structure from {nemo_model_file} to {nest_model_file}")
        nemo_structure = torch.load(nemo_model_file, map_location='cpu', weights_only=False)
        torch.save(nemo_structure, nest_model_file)
        print(f"  Copied {len(nemo_structure)} parameters")
    else:
        print(f"NeMo model structure file not found: {nemo_model_file}")
        return False
    
    print("Weight synchronization complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Sync weights from NeMo to nest_ssl_project')
    parser.add_argument('--nemo_dir', type=str, 
                       default='nest_ssl_project/saved_nemo_outputs',
                       help='Directory containing NeMo saved outputs')
    parser.add_argument('--nest_dir', type=str,
                       default='nest_ssl_project/saved_nest_outputs',
                       help='Directory to save nest outputs')
    parser.add_argument('--step', type=str, default=None,
                       help='Step to sync (e.g., "step_0"). Auto-detected if not provided.')
    
    args = parser.parse_args()
    
    success = sync_weights(args.nemo_dir, args.nest_dir, args.step)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

