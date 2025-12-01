# Checkpoint Compatibility Verification

This script verifies that checkpoints saved by the local implementation can be correctly loaded back.

## Usage

### Basic Usage

```bash
# Test loading checkpoint
python tools/verify_checkpoint_compatibility.py --ckpt_path <path_to_checkpoint>

# Test loading with custom config
python tools/verify_checkpoint_compatibility.py --ckpt_path checkpoint.nemo --config config/nest_fast-conformer.yaml

# Test save and reload round-trip
python tools/verify_checkpoint_compatibility.py --ckpt_path checkpoint.nemo --test_save_load
```

## Features

- **Load checkpoint**: Tests loading `.nemo` or `.ckpt` files
- **Save/reload test**: Optionally tests save and reload round-trip to verify checkpoint integrity
- **State dict comparison**: Compares original and reloaded state dicts to ensure consistency

## Output

The script will:
1. Load the checkpoint in the local model
2. Print model information (parameter counts, etc.)
3. If `--test_save_load` is used, save and reload the model, then compare state dicts
4. Report any mismatches in keys, shapes, or values

## Example Output

```
============================================================
Checkpoint Loading Verification
============================================================
Checkpoint: /path/to/checkpoint.nemo
Test save/reload: True
============================================================

############################################################
TEST: Loading checkpoint in local model
############################################################

============================================================
Loading LOCAL model from checkpoint: /path/to/checkpoint.nemo
============================================================
✓ Successfully loaded .nemo checkpoint
  Total parameters: 123,456,789
  Trainable parameters: 123,456,789

✓ SUCCESS: Checkpoint can be loaded in local model

============================================================
Testing save and reload round-trip
============================================================
✓ Model saved to /path/to/checkpoint.test.nemo

============================================================
Loading LOCAL model from checkpoint: /path/to/checkpoint.test.nemo
============================================================
✓ Successfully loaded .nemo checkpoint
  Total parameters: 123,456,789
  Trainable parameters: 123,456,789

============================================================
Comparing state dicts: Original Model vs Reloaded Model
============================================================

Keys comparison:
  Common keys: 150
  Only in Original Model: 0
  Only in Reloaded Model: 0

Shape mismatches: 0

Value mismatches: 0

✓ State dicts are identical!

✓ SUCCESS: Save and reload round-trip works correctly!
```

## Requirements

- PyTorch
- PyTorch Lightning
- OmegaConf

## Notes

- The script supports both `.nemo` files and PyTorch Lightning `.ckpt` files
- State dict comparison uses `torch.allclose` with `atol=1e-5` tolerance
- When using `--test_save_load`, a test file will be created with `.test.nemo` suffix

