# Checkpoint Compatibility Verification

This script verifies that checkpoints saved by the local implementation can be loaded by NeMo, and vice versa.

## Usage

### Basic Usage

```bash
# Test bidirectional loading (default)
python tools/verify_checkpoint_compatibility.py --ckpt_path <path_to_checkpoint>

# Test loading NeMo checkpoint in local model
python tools/verify_checkpoint_compatibility.py --ckpt_path nemo_checkpoint.nemo --direction nemo_to_local

# Test loading local checkpoint in NeMo model
python tools/verify_checkpoint_compatibility.py --ckpt_path local_checkpoint.nemo --direction local_to_nemo

# Test bidirectional with custom config
python tools/verify_checkpoint_compatibility.py --ckpt_path checkpoint.nemo --config config/nest_fast-conformer.yaml
```

## Test Directions

- **`bidirectional`** (default): Loads checkpoint in both local and NeMo models, then compares state dicts
- **`nemo_to_local`**: Tests loading NeMo checkpoint in local model
- **`local_to_nemo`**: Tests loading local checkpoint in NeMo model (requires NeMo to be installed)

## Output

The script will:
1. Load the checkpoint in the specified model(s)
2. Print model information (parameter counts, etc.)
3. Compare state dicts if both models are loaded
4. Report any mismatches in keys, shapes, or values

## Example Output

```
============================================================
Checkpoint Compatibility Verification
============================================================
Checkpoint: /path/to/checkpoint.nemo
Direction: bidirectional
NeMo available: True
============================================================

############################################################
BIDIRECTIONAL TEST: Loading checkpoint in both models and comparing
############################################################

============================================================
Loading LOCAL model from checkpoint: /path/to/checkpoint.nemo
============================================================
✓ Successfully loaded .nemo checkpoint
  Total parameters: 123,456,789
  Trainable parameters: 123,456,789

============================================================
Loading NEMO model from checkpoint: /path/to/checkpoint.nemo
============================================================
✓ Successfully loaded as EncDecDenoiseMaskedTokenPredModel
  Total parameters: 123,456,789
  Trainable parameters: 123,456,789

============================================================
Comparing state dicts: Local Model vs NeMo Model
============================================================

Keys comparison:
  Common keys: 150
  Only in Local Model: 0
  Only in NeMo Model: 0

Shape mismatches: 0

Value mismatches: 0

✓ State dicts are identical!

✓ SUCCESS: Checkpoints are fully compatible!
```

## Requirements

- PyTorch
- PyTorch Lightning
- OmegaConf
- (Optional) NeMo for bidirectional testing

## Notes

- The script supports both `.nemo` files and PyTorch Lightning `.ckpt` files
- If NeMo is not available, only local model loading will be tested
- State dict comparison uses `torch.allclose` with `atol=1e-5` tolerance

