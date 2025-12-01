# Intermediate Output Comparison

This script compares intermediate layer outputs between NeMo and nest_ssl_project implementations to ensure they produce identical results under the same conditions.

## Features

- **Same seed**: Sets identical random seed for both models
- **Same checkpoint**: Loads the same checkpoint weights (or compatible checkpoints)
- **Same input**: Uses identical test input data
- **Layer-by-layer comparison**: Captures and compares outputs from every layer
- **Detailed reporting**: Shows which layers match and which differ

## Usage

### Basic Usage

```bash
# Compare with same checkpoint
python tools/compare_intermediate_outputs.py \
    --ckpt_path local_checkpoint.nemo \
    --nemo_ckpt_path nemo_checkpoint.nemo \
    --seed 42

# Custom test input
python tools/compare_intermediate_outputs.py \
    --ckpt_path local_checkpoint.nemo \
    --nemo_ckpt_path nemo_checkpoint.nemo \
    --seed 42 \
    --batch_size 4 \
    --seq_len 32000

# Adjust tolerance
python tools/compare_intermediate_outputs.py \
    --ckpt_path local_checkpoint.nemo \
    --nemo_ckpt_path nemo_checkpoint.nemo \
    --atol 1e-6 \
    --rtol 1e-6
```

## Parameters

- `--ckpt_path`: Path to local checkpoint file (.nemo or .ckpt)
- `--nemo_ckpt_path`: Path to NeMo checkpoint file (.nemo or .ckpt)
- `--seed`: Random seed for reproducibility (default: 42)
- `--batch_size`: Batch size for test input (default: 2)
- `--seq_len`: Sequence length for test input (default: 16000)
- `--atol`: Absolute tolerance for comparison (default: 1e-5)
- `--rtol`: Relative tolerance for comparison (default: 1e-5)
- `--config`: Path to model config file (optional)

## Output

The script provides:

1. **Final Output Comparison**: Compares the final model outputs
2. **Layer Statistics**: Shows how many layers are common, only in local, or only in NeMo
3. **Layer-by-Layer Comparison**: Table showing match status, max difference, and shape for each layer
4. **Mismatched Layers Details**: Detailed information about layers that don't match

## Example Output

```
================================================================================
Intermediate Output Comparison
================================================================================
Local checkpoint: local_checkpoint.nemo
NeMo checkpoint: nemo_checkpoint.nemo
Seed: 42
Batch size: 2
Sequence length: 16000
Tolerance: atol=1e-05, rtol=1e-05
================================================================================

Loading local model...
Loading NeMo model...

Creating test input (seed=42)...
Input shape: torch.Size([2, 16000])
Input length: tensor([16000, 16000])

Registering hooks on local model...
Registering hooks on NeMo model...

Running forward pass on local model...
Running forward pass on NeMo model...

================================================================================
Comparing Final Outputs
================================================================================
Final output match: True

================================================================================
Comparing Intermediate Layer Outputs
================================================================================

Layer statistics:
  Common layers: 150
  Only in local: 5
  Only in NeMo: 3

Layer Name                                              Match      Max Abs Diff     Shape               
---------------------------------------------------------------------------------------------------------
encoder.layers.0.conv_module.batch_norm                ✓          0.00e+00          torch.Size([2, 512, 4000])
encoder.layers.0.conv_module.depthwise_conv            ✓          0.00e+00          torch.Size([2, 512, 4000])
encoder.layers.0.feed_forward.linear1                  ✓          0.00e+00          torch.Size([2, 4000, 2048])
...

================================================================================
Summary
================================================================================
Total common layers compared: 150
Matched layers: 148
Mismatched layers: 2
Match rate: 98.67%
```

## How It Works

1. **Seed Setting**: Sets identical random seed for PyTorch, NumPy, and CUDA
2. **Model Loading**: Loads both models from checkpoints
3. **Hook Registration**: Registers forward hooks on all modules to capture intermediate outputs
4. **Forward Pass**: Runs forward pass on both models with identical input
5. **Output Capture**: Captures outputs from each layer during forward pass
6. **Comparison**: Compares outputs layer by layer using `torch.allclose`
7. **Reporting**: Generates detailed report of matches and mismatches

## Requirements

- PyTorch
- PyTorch Lightning
- OmegaConf
- NeMo (for NeMo model loading)

## Notes

- The script uses `torch.no_grad()` to disable gradient computation for efficiency
- Hooks are automatically cleaned up after comparison
- Layer names are normalized (prefixes removed) for comparison
- Both tuple and tensor outputs are supported

## Troubleshooting

### "NeMo is not available"
Install NeMo or ensure it's in your Python path.

### "Shape mismatch" errors
This indicates the models have different architectures. Check that you're comparing compatible models.

### Large differences in early layers
If early layers (preprocessor, encoder start) show large differences, check:
- Preprocessing configuration
- Input normalization
- Feature extraction settings

### Differences only in specific layers
If only certain layers differ:
- Check layer-specific implementations
- Verify activation functions
- Check normalization layers (BatchNorm, LayerNorm)

