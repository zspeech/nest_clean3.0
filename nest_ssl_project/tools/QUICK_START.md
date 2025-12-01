# Quick Start - Complete Commands

## NeMo Training with Output Saver (Complete Command)

### Using Default Dummy Data

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

### Windows PowerShell (Single Line)

```powershell
python tools/nemo_training_with_saver.py --config-path ../conf/ssl/nest --config-name nest_fast-conformer output_dir=./saved_nemo_outputs seed=42 save_steps="0,1,2,3,4" model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

### Using Your Own Data

Replace the manifest paths with your actual paths:

```bash
python tools/nemo_training_with_saver.py \
    --config-path ../conf/ssl/nest \
    --config-name nest_fast-conformer \
    output_dir=./saved_nemo_outputs \
    seed=42 \
    save_steps="0,1,2,3,4" \
    model.train_ds.manifest_filepath=/absolute/path/to/your/train.json \
    model.train_ds.noise_manifest=/absolute/path/to/your/noise.json \
    model.validation_ds.manifest_filepath=/absolute/path/to/your/val.json \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=5 \
    trainer.num_sanity_val_steps=0
```

## nest_ssl_project Training with Comparator (Complete Command)

### Using Default Dummy Data

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

## Notes

1. **Path Format**: Use forward slashes `/` even on Windows, or use relative paths like `nest_ssl_project/data/dummy_ssl/train_manifest.json`
2. **Same Seed**: Both commands must use `seed=42` (or same value)
3. **Same Data**: Use the same manifest files in both commands for fair comparison
4. **Quick Test**: Set `trainer.limit_train_batches=5` to test with just 5 batches

