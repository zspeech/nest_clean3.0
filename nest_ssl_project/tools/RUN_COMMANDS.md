# Complete Run Commands

## NeMo Training with Output Saver

### Basic Command (Save First 5 Steps)

```bash
python tools/nemo_training_with_saver.py \
    --config-path ../conf/ssl/nest \
    --config-name nest_fast-conformer \
    output_dir=./saved_nemo_outputs \
    seed=42 \
    save_steps="0,1,2,3,4" \
    model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json \
    model.train_ds.noise_manifest=null \
    model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=5 \
    trainer.num_sanity_val_steps=0
```

### With Custom Data Paths

```bash
python tools/nemo_training_with_saver.py \
    --config-path ../conf/ssl/nest \
    --config-name nest_fast-conformer \
    output_dir=./saved_nemo_outputs \
    seed=42 \
    save_steps="0,1,2,3,4,10,20" \
    model.train_ds.manifest_filepath=/path/to/your/train.json \
    model.train_ds.noise_manifest=/path/to/your/noise.json \
    model.validation_ds.manifest_filepath=/path/to/your/val.json \
    model.train_ds.batch_size=4 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=20 \
    trainer.num_sanity_val_steps=0
```

### Windows PowerShell (Single Line)

```powershell
python tools/nemo_training_with_saver.py --config-path ../conf/ssl/nest --config-name nest_fast-conformer output_dir=./saved_nemo_outputs seed=42 save_steps="0,1,2,3,4" model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

## nest_ssl_project Training with Comparator

### Basic Command

```bash
python tools/nest_training_with_comparator.py \
    --config-path config \
    --config-name nest_fast-conformer \
    saved_outputs_dir=./saved_nemo_outputs \
    comparison_output_dir=./comparison_results \
    seed=42 \
    atol=1e-5 \
    rtol=1e-5 \
    model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json \
    model.train_ds.noise_manifest=null \
    model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=5 \
    trainer.num_sanity_val_steps=0
```

### Windows PowerShell (Single Line)

```powershell
python tools/nest_training_with_comparator.py --config-path config --config-name nest_fast-conformer saved_outputs_dir=./saved_nemo_outputs comparison_output_dir=./comparison_results seed=42 atol=1e-5 rtol=1e-5 model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

## Quick Reference

### Key Parameters

- `output_dir`: Where to save NeMo outputs (required for saver)
- `saved_outputs_dir`: Where saved NeMo outputs are (required for comparator)
- `seed`: Must be the same for both NeMo and nest_ssl_project (default: 42)
- `save_steps`: Which steps to save (e.g., "0,1,2,3,4")
- `trainer.limit_train_batches`: Limit number of batches (for quick testing)

### Important Notes

1. **Same seed**: Both commands must use the same `seed` value
2. **Same data**: Use the same manifest files for fair comparison
3. **Same batch size**: Use same `model.train_ds.batch_size` in both
4. **Step matching**: `save_steps` should match steps that will be compared

