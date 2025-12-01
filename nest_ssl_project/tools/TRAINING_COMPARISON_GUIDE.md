# Training Comparison Guide

This guide explains how to compare NeMo and nest_ssl_project training outputs step by step.

## Workflow Overview

1. **Step 1**: Run NeMo training with output saver (saves outputs for specified steps)
2. **Step 2**: Run nest_ssl_project training with comparator (compares with saved outputs)

## Step 1: Save NeMo Training Outputs

### Option A: Use Modified Training Script

Copy `training_output_saver.py` to your NeMo environment, then modify NeMo's training script:

```python
from tools.training_output_saver import TrainingOutputSaver

# In your NeMo training script
saver = TrainingOutputSaver(
    output_dir="./saved_nemo_outputs",
    seed=42,
    save_every_n_steps=1,  # Save every step, or use save_steps list
)
saver.setup_hooks(asr_model)
saver.save_model_structure(asr_model)

# During training loop (if not using Lightning)
for batch_idx, batch in enumerate(train_dataloader):
    output = asr_model(batch)
    loss = compute_loss(output)
    loss.backward()
    
    saver.save_step(
        step=batch_idx,
        batch=batch,
        forward_output=output,
        loss=loss,
    )
    
    optimizer.step()
    optimizer.zero_grad()

saver.finalize()
```

### Option B: Use PyTorch Lightning Callback

Use the provided callback wrapper:

```python
from tools.training_output_saver import TrainingOutputSaverCallback

# In your NeMo training script
saver_callback = TrainingOutputSaverCallback(
    output_dir="./saved_nemo_outputs",
    seed=42,
    save_steps=[0, 1, 2, 3, 4, 10, 20, 50],  # Specific steps to save
)
trainer.callbacks.append(saver_callback)
```

### Option C: Use Standalone Script (if available)

```bash
# In NeMo environment
python tools/nemo_training_with_saver.py \
    --config-path ../conf/ssl/nest \
    --config-name nest_fast-conformer \
    --output_dir ./saved_nemo_outputs \
    --seed 42 \
    --save_steps "0,1,2,3,4,10,20,50" \
    model.train_ds.manifest_filepath=<path> \
    trainer.devices=1 \
    trainer.max_epochs=1
```

## Step 2: Compare with nest_ssl_project Training

### Option A: Use Modified Training Script

Modify `train.py` to include comparator:

```python
from tools.training_output_saver import TrainingOutputComparator, ForwardBackwardHook

# In train.py
comparator = TrainingOutputComparator(
    saved_outputs_dir="./saved_nemo_outputs",
    comparison_output_dir="./comparison_results",
    atol=1e-5,
    rtol=1e-5,
)

# Register hooks
hooks = {}
for name, module in asr_model.named_modules():
    if name:
        hooks[name] = ForwardBackwardHook(name)
        hooks[name].register(module)

# During training (if not using Lightning)
for batch_idx, batch in enumerate(train_dataloader):
    output = asr_model(batch)
    loss = compute_loss(output)
    loss.backward()
    
    comparison_result = comparator.compare_step(
        step=batch_idx,
        batch=batch,
        forward_output=output,
        loss=loss,
        model=asr_model,
        hooks=hooks,
    )
    
    # Print result
    if comparison_result.get('forward_output_match', {}).get('match', False):
        print(f"Step {batch_idx}: ✓ Match")
    else:
        print(f"Step {batch_idx}: ✗ Mismatch")

comparator.print_summary()
```

### Option B: Use PyTorch Lightning Callback

Use the provided callback wrapper:

```python
from tools.training_output_saver import TrainingOutputComparatorCallback

# In train.py
comparator_callback = TrainingOutputComparatorCallback(
    saved_outputs_dir="./saved_nemo_outputs",
    comparison_output_dir="./comparison_results",
    seed=42,  # Must match NeMo training seed
    atol=1e-5,
    rtol=1e-5,
)
trainer.callbacks.append(comparator_callback)
```

### Option C: Use Standalone Script

```bash
# In nest_ssl_project environment
python tools/nest_training_with_comparator.py \
    --config-path config \
    --config-name nest_fast-conformer \
    --saved_outputs_dir ./saved_nemo_outputs \
    --comparison_output_dir ./comparison_results \
    --seed 42 \
    model.train_ds.manifest_filepath=<path> \
    trainer.devices=1 \
    trainer.max_epochs=1
```

## Important Requirements

### 1. Same Seed
Both training runs **must** use the same seed:

```python
# In both NeMo and nest_ssl_project
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
```

### 2. Same Batch Order
Ensure data loaders produce the same batch order:
- Use same `shuffle=False` or same `shuffle=True` with same seed
- Use same `batch_size`
- Use same dataset (or at least same samples for comparison steps)

### 3. Same Model Configuration
- Same architecture (encoder, decoder, etc.)
- Same initialization (same checkpoint or same random init with same seed)

### 4. Same Training Mode
- Both models should be in training mode (`model.train()`)
- Same dropout settings
- Same batch normalization settings

## Saved Output Structure

```
saved_nemo_outputs/
├── metadata.pkl                    # Global metadata (seed, saved steps, etc.)
├── model_structure.pkl            # Model structure information
└── step_<N>/
    ├── batch.pt                   # Input batch
    ├── forward_output.pt          # Forward output
    ├── loss.pt                    # Loss value
    ├── parameter_gradients.pt     # Parameter gradients
    ├── hook_data.pkl             # Intermediate layer outputs
    └── metadata.pkl               # Step-specific metadata
```

## Comparison Output Structure

```
comparison_results/
└── comparison_step_<N>.pkl       # Comparison results for each step
```

## Example: Complete Workflow

### 1. NeMo Environment

```bash
# Activate NeMo environment
conda activate nemo

# Copy training_output_saver.py to NeMo tools directory
cp nest_ssl_project/tools/training_output_saver.py NeMo/tools/

# Run training with saver
python examples/asr/speech_pretraining/masked_token_pred_pretrain.py \
    --config-path ../conf/ssl/nest \
    --config-name nest_fast-conformer \
    model.train_ds.manifest_filepath=/path/to/train.json \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10 \
    # Add saver callback in code or use wrapper script
```

### 2. nest_ssl_project Environment

```bash
# Activate nest_ssl_project environment
conda activate nest_ssl

# Run training with comparator
python train.py \
    --config-path config \
    --config-name nest_fast-conformer \
    --saved_outputs_dir ./saved_nemo_outputs \
    --comparison_output_dir ./comparison_results \
    --seed 42 \
    model.train_ds.manifest_filepath=/path/to/train.json \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=10
```

## Troubleshooting

### "Step N not found in saved outputs"
- Ensure you're comparing the same step numbers
- Check that NeMo training saved outputs for those steps

### "Shape mismatch" errors
- Verify models have the same architecture
- Check that batch sizes match

### Large differences in outputs
- Verify same seed is used
- Check that data loaders produce same batches
- Ensure models are initialized identically
- Verify training mode settings match

### Memory issues
- Reduce number of steps to save
- Use smaller batch size
- Save outputs to CPU instead of GPU

## Advanced Usage

### Save Only Specific Steps

```python
saver = TrainingOutputSaver(
    output_dir="./saved_nemo_outputs",
    seed=42,
    save_every_n_steps=999999,  # Don't save by default
    save_first_n_steps=5,  # But save first 5 steps
)
# Then manually save specific steps:
saver.save_step(step=100, ...)
saver.save_step(step=200, ...)
```

### Compare Only Specific Steps

The comparator will automatically skip steps that don't exist in saved outputs.

### Custom Comparison Logic

Extend `TrainingOutputComparator` class to add custom comparison logic:

```python
class CustomComparator(TrainingOutputComparator):
    def compare_step(self, ...):
        result = super().compare_step(...)
        # Add custom comparison logic
        return result
```

