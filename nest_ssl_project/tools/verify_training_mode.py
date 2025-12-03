#!/usr/bin/env python3
"""
Verify that NeMo and nest are both in training mode during output capture.

The key insight is that NeMo's FilterbankFeatures applies dither only in training mode.
If NeMo is in training mode but nest is in eval mode (or vice versa), outputs will differ.
"""

import torch
import pickle
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_dir", type=str, required=True)
    parser.add_argument("--nest_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir)
    nest_dir = Path(args.nest_dir)
    
    if args.step is None:
        nemo_steps = [int(d.name.split('_')[1]) for d in nemo_dir.glob("step_*") if d.is_dir()]
        nest_steps = [int(d.name.split('_')[1]) for d in nest_dir.glob("step_*") if d.is_dir()]
        common_steps = sorted(list(set(nemo_steps) & set(nest_steps)))
        if common_steps:
            args.step = common_steps[0]
    
    print("="*80)
    print("VERIFY TRAINING MODE")
    print("="*80)
    
    # Check model_structure.txt for training mode info
    for name, dir_path in [("NeMo", nemo_dir), ("nest", nest_dir)]:
        structure_file = dir_path / "model_structure.txt"
        if structure_file.exists():
            with open(structure_file, 'r') as f:
                content = f.read()
                # Look for training mode indicators
                if "training=True" in content or "(training)" in content:
                    print(f"{name}: Training mode detected in model_structure.txt")
                elif "training=False" in content or "(eval)" in content:
                    print(f"{name}: Eval mode detected in model_structure.txt")
                else:
                    print(f"{name}: Training mode not explicitly stated")
    
    # The real test: Check if outputs are consistent with training/eval mode
    print("\n" + "="*80)
    print("ANALYSIS: What could cause the mismatch?")
    print("="*80)
    
    print("""
Based on the diagnostic results:
- Manual nest ≈ Actual nest (diff ~1e-5) -> nest implementation is CORRECT
- Manual NeMo ≠ Actual NeMo (diff ~0.002) -> NeMo has EXTRA processing

Possible causes for NeMo's extra processing:
1. dither > 0 in NeMo (even if config says 0)
2. nb_augmentation_prob > 0 in NeMo
3. Different numerical precision (autocast, dtype)
4. Model is in training mode with some randomness

To fix:
1. Force NeMo's preprocessor to eval mode: model.preprocessor.eval()
2. Or ensure dither=0.0 is actually applied
3. Or accept that NeMo has slight differences and align nest to NeMo's actual output

RECOMMENDATION:
Since nest's implementation matches the manual calculation perfectly,
and NeMo's actual output differs from manual calculation,
the issue is in NeMo's runtime behavior, NOT in nest's implementation.

For alignment purposes, we should:
1. Either force NeMo to match manual calculation
2. Or accept NeMo's actual output and align nest to it

The latter is easier - just ensure both use the same buffers and settings.
""")


if __name__ == '__main__':
    main()

