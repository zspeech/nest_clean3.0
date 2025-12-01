# Saved Outputs Comparison

This workflow allows you to compare NeMo and nest_ssl_project implementations by:
1. Running NeMo model in NeMo environment and saving all outputs
2. Running nest_ssl_project model with same seed and comparing with saved outputs

This approach avoids environment conflicts and allows for detailed comparison.

## Workflow

### Step 1: Save NeMo Outputs

Run this in NeMo environment:

```bash
# In NeMo environment
python tools/save_nemo_outputs.py \
    --ckpt_path nemo_checkpoint.nemo \
    --output_dir ./saved_nemo_outputs \
    --seed 42 \
    --batch_size 2 \
    --seq_len 16000
```

This will save:
- `input_data.pt`: Input audio signal and lengths
- `forward_output.pt`: Final forward output
- `parameter_gradients.pt`: Parameter gradients after backward pass
- `hook_data.pkl`: Intermediate layer outputs (forward and backward)
- `metadata.pkl`: Metadata (seed, batch size, etc.)

### Step 2: Compare with nest_ssl_project

Run this in nest_ssl_project environment:

```bash
# In nest_ssl_project environment
python tools/compare_with_saved_outputs.py \
    --ckpt_path local_checkpoint.nemo \
    --saved_outputs_dir ./saved_nemo_outputs \
    --atol 1e-5 \
    --rtol 1e-5
```

This will:
- Load the saved input data
- Set the same seed
- Run forward and backward passes
- Compare all outputs with saved NeMo outputs

## What Gets Compared

### 1. Final Forward Output
- Compares the final model output after forward pass

### 2. Parameter Gradients
- Compares gradients for all model parameters after backward pass
- Shows which parameters have matching gradients

### 3. Intermediate Layer Outputs
- **Forward outputs**: Outputs from each layer during forward pass
- **Backward gradients**: Gradients w.r.t. layer outputs during backward pass

## Output Format

The comparison script provides:

1. **Final Forward Output Comparison**: Shows if final outputs match
2. **Parameter Gradients Table**: Lists all parameters and their gradient match status
3. **Intermediate Layers Table**: Shows forward and backward match status for each layer
4. **Summary**: Overall statistics and match rates

## Example Output

```
================================================================================
Comparing Final Forward Output
================================================================================
Forward output match: True

================================================================================
Comparing Parameter Gradients
================================================================================
Common parameters: 200

Parameter Name                                          Match      Max Abs Diff
-------------------------------------------------------------------------------------
encoder.layers.0.conv_module.batch_norm.weight         ✓          0.00e+00
encoder.layers.0.conv_module.depthwise_conv.weight      ✓          0.00e+00
...

================================================================================
Comparing Intermediate Layer Outputs
================================================================================

Common layers: 150

Layer Name                                              Forward Match   Backward Match   Max Diff
---------------------------------------------------------------------------------------------------------
encoder.layers.0.conv_module.batch_norm                 ✓               ✓               0.00e+00
encoder.layers.0.conv_module.depthwise_conv             ✓               ✓               0.00e+00
...

================================================================================
Summary
================================================================================
Forward output match: True

Parameter gradients:
  Matched: 200
  Mismatched: 0
  Match rate: 100.00%

Intermediate layers:
  Forward matches: 150
  Forward mismatches: 0
  Backward matches: 150
  Backward mismatches: 0
  Forward match rate: 100.00%
  Backward match rate: 100.00%

✓ All outputs match!
```

## Parameters

### save_nemo_outputs.py

- `--ckpt_path`: Path to NeMo checkpoint (.nemo or .ckpt)
- `--output_dir`: Directory to save outputs
- `--seed`: Random seed (default: 42)
- `--batch_size`: Batch size (default: 2)
- `--seq_len`: Sequence length (default: 16000)

### compare_with_saved_outputs.py

- `--ckpt_path`: Path to local checkpoint (.nemo or .ckpt)
- `--saved_outputs_dir`: Directory containing saved NeMo outputs
- `--atol`: Absolute tolerance (default: 1e-5)
- `--rtol`: Relative tolerance (default: 1e-5)
- `--config`: Path to model config (optional)

## Requirements

### For save_nemo_outputs.py
- NeMo environment
- PyTorch
- Checkpoint file

### For compare_with_saved_outputs.py
- nest_ssl_project environment
- PyTorch
- Same checkpoint (or compatible checkpoint)

## Notes

- **Same seed**: Both scripts use the same seed to ensure reproducibility
- **Same input**: The comparison script loads the exact input used in NeMo
- **Gradient comparison**: Requires training mode (`model.train()`) to compute gradients
- **Memory**: Saving all intermediate outputs can use significant memory/disk space
- **Compatibility**: Checkpoints should be compatible (same architecture)

## Troubleshooting

### "Shape mismatch" errors
- Check that models have the same architecture
- Verify checkpoint compatibility

### Large differences in early layers
- Check preprocessing configuration
- Verify input normalization

### Gradient mismatches
- Ensure both models are in training mode
- Check that loss computation is identical
- Verify dropout and other stochastic layers are disabled or use same seed

### Memory issues
- Reduce batch size or sequence length
- Save outputs in smaller chunks
- Use CPU for comparison (move tensors to CPU)

## Advantages of This Approach

1. **No environment conflicts**: Run NeMo and nest_ssl_project separately
2. **Reproducible**: Same seed ensures identical conditions
3. **Detailed**: Captures forward and backward outputs
4. **Reusable**: Saved outputs can be compared multiple times
5. **Flexible**: Can compare with different checkpoints or configurations

