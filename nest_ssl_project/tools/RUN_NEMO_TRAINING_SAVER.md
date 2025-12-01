# How to Run NeMo Training with Output Saver

## Prerequisites

1. **NeMo Environment**: This script must be run in NeMo environment
2. **Copy training_output_saver.py**: Copy `nest_ssl_project/tools/training_output_saver.py` to NeMo's tools directory, or ensure it's in Python path

## Basic Usage

```bash
# In NeMo environment
python tools/nemo_training_with_saver.py \
    --config-path ../conf/ssl/nest \
    --config-name nest_fast-conformer \
    --output_dir ./saved_nemo_outputs \
    --seed 42 \
    --save_steps "0,1,2,3,4,10,20,50" \
    model.train_ds.manifest_filepath=/path/to/train.json \
    model.train_ds.noise_manifest=/path/to/noise.json \
    model.validation_ds.manifest_filepath=/path/to/val.json \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10
```

## Required Arguments

- `--output_dir`: Directory to save outputs (required)
- `--seed`: Random seed (default: 42)
- `--save_steps`: Comma-separated list of steps to save (e.g., "0,1,2,3,4,10,20,50")
  - If not specified, saves all steps (may use lots of disk space)
  - Recommended: save first few steps and some later steps for comparison

## Optional Arguments

- `--config-path`: Config path (default: `../conf/ssl/nest`)
- `--config-name`: Config name (default: `nest_fast-conformer`)

## Additional Training Arguments

You can pass any standard NeMo training arguments:

```bash
python tools/nemo_training_with_saver.py \
    --output_dir ./saved_nemo_outputs \
    --seed 42 \
    --save_steps "0,1,2,3,4" \
    model.train_ds.manifest_filepath=/path/to/train.json \
    model.train_ds.noise_manifest=/path/to/noise.json \
    model.train_ds.batch_size=4 \
    model.validation_ds.manifest_filepath=/path/to/val.json \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10 \
    trainer.num_sanity_val_steps=0
```

## Example: Save First 5 Steps Only

```bash
python tools/nemo_training_with_saver.py \
    --output_dir ./saved_nemo_outputs \
    --seed 42 \
    --save_steps "0,1,2,3,4" \
    model.train_ds.manifest_filepath=/path/to/train.json \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=5
```

## Example: Save Every 10th Step

```bash
# Modify script to use save_every_n_steps=10, or save specific steps:
python tools/nemo_training_with_saver.py \
    --output_dir ./saved_nemo_outputs \
    --seed 42 \
    --save_steps "0,10,20,30,40,50" \
    model.train_ds.manifest_filepath=/path/to/train.json \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=50
```

## Output Structure

After running, you'll have:

```
saved_nemo_outputs/
├── metadata.pkl                    # Global metadata
├── model_structure.pkl            # Model structure info
└── step_<N>/
    ├── batch.pt                   # Input batch
    ├── forward_output.pt          # Forward output
    ├── loss.pt                   # Loss value
    ├── parameter_gradients.pt     # Parameter gradients
    ├── hook_data.pkl             # Intermediate outputs
    └── metadata.pkl              # Step metadata
```

## Troubleshooting

### ImportError: Cannot import TrainingOutputSaver

**Solution**: Copy `training_output_saver.py` to NeMo environment:

```bash
# Copy the file
cp nest_ssl_project/tools/training_output_saver.py NeMo/tools/

# Or add to Python path
export PYTHONPATH=$PYTHONPATH:/path/to/nest_ssl_project
```

### Hydra config not found

**Solution**: Check that config path is correct:

```bash
# List available configs
ls NeMo/examples/asr/conf/ssl/nest/

# Use correct path
--config-path examples/asr/conf/ssl/nest
```

### Out of memory

**Solution**: Reduce number of steps to save or use smaller batch size:

```bash
--save_steps "0,1,2"  # Save only first 3 steps
model.train_ds.batch_size=2  # Smaller batch
```

### No outputs saved

**Check**:
1. Training actually ran (check logs)
2. Steps specified in `--save_steps` match actual training steps
3. Output directory is writable

## Next Steps

After saving outputs, use `nest_training_with_comparator.py` to compare:

```bash
# In nest_ssl_project environment
python tools/nest_training_with_comparator.py \
    --saved_outputs_dir ./saved_nemo_outputs \
    --seed 42 \
    ...
```

