# Ready-to-Run Commands

## NeMo Training with Output Saver

**Note**: Default values are set in config file (`nest_fast-conformer.yaml`). You can override them via command line if needed.

### Windows PowerShell (Copy and Run - Single Line)

```powershell
# Using config file defaults (output_dir, seed, save_steps are already set in config)
# From NeMo root directory, if you copied nemo_training_with_saver.py to NeMo/tools/
python tools/nemo_training_with_saver.py --config-path examples/asr/conf/ssl/nest --config-name nest_fast-conformer model.train_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0

# Or from nest_ssl_project directory with absolute path
python tools/nemo_training_with_saver.py --config-path C:/Users/zhile/Desktop/Nemo_nest/NeMo/examples/asr/conf/ssl/nest --config-name nest_fast-conformer model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

### Linux/Mac Bash (Copy and Run - Multi-line)

```bash
# Using config file defaults
python tools/nemo_training_with_saver.py \
    --config-path examples/asr/conf/ssl/nest \
    --config-name nest_fast-conformer \
    model.train_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/train_manifest.json \
    model.train_ds.noise_manifest=null \
    model.validation_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/val_manifest.json \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=5 \
    trainer.num_sanity_val_steps=0
```

### Windows CMD (Copy and Run - Single Line)

```cmd
python tools/nemo_training_with_saver.py --config-path C:/Users/zhile/Desktop/Nemo_nest/NeMo/examples/asr/conf/ssl/nest --config-name nest_fast-conformer model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

## nest_ssl_project Training with Comparator

**Note**: Default values are set in config file (`nest_fast-conformer.yaml`). You can override them via command line if needed.

### Windows PowerShell (Copy and Run - Single Line)

```powershell
# Using config file defaults (saved_outputs_dir, comparison_output_dir, seed, atol, rtol are already set in config)
python tools/nest_training_with_comparator.py --config-path config --config-name nest_fast-conformer model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

### Linux/Mac Bash (Copy and Run - Multi-line)

```bash
# Using config file defaults
python tools/nest_training_with_comparator.py \
    --config-path config \
    --config-name nest_fast-conformer \
    model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json \
    model.train_ds.noise_manifest=null \
    model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json \
    trainer.devices=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=5 \
    trainer.num_sanity_val_steps=0
```

### Windows CMD (Copy and Run - Single Line)

```cmd
python tools/nest_training_with_comparator.py --config-path config --config-name nest_fast-conformer model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

## Notes

- **Config file defaults**: All output saver/comparator parameters are set in `nest_fast-conformer.yaml`
  - `output_dir: ./saved_nemo_outputs`
  - `seed: 42`
  - `save_steps: "0,1,2,3,4"`
  - `saved_outputs_dir: ./saved_nemo_outputs`
  - `comparison_output_dir: ./comparison_results`
  - `atol: 1e-5`, `rtol: 1e-5`
- You can override these values via command line if needed (e.g., `output_dir=./custom_output seed=100`)
- All paths use relative paths from project root
- Uses dummy data from `nest_ssl_project/data/dummy_ssl/`
- Limits training to 5 batches for quick testing
- Both commands use same seed=42 for fair comparison

