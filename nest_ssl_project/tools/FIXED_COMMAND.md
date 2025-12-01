# Fixed Command - Correct Hydra Syntax

## Problem
- Hydra struct mode: Use `+key=value` to add NEW keys not in config file
- If key exists in config: Use `key=value` to override (no + prefix)
- If key exists but want to force override: Use `++key=value`

## Fixed Commands

### NeMo Training with Output Saver

```powershell
# limit_train_batches: use + (new key)
# num_sanity_val_steps: use without + (exists in config, just override)
python tools/nemo_training_with_saver.py --config-path C:/Users/zhile/Desktop/Nemo_nest/NeMo/examples/asr/conf/ssl/nest --config-name nest_fast-conformer model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 +trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

### nest_ssl_project Training with Comparator

```powershell
# limit_train_batches: use + (new key)
# num_sanity_val_steps: use without + (exists in config, just override)
python tools/nest_training_with_comparator.py --config-path config --config-name nest_fast-conformer model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 +trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

## Key Points

- `+key=value`: Add NEW key not in config file
- `key=value`: Override existing key in config file
- `++key=value`: Force override even if key exists (use sparingly)
- Config file defaults (output_dir, seed, etc.) are already set, no need to specify

