# Fixed Command - Use + Prefix for New Keys

## Problem
Hydra struct mode doesn't allow adding new keys. Use `+` prefix to add new configuration keys.

## Fixed Commands

### NeMo Training with Output Saver

```powershell
# Use + prefix for keys not in config file
python tools/nemo_training_with_saver.py --config-path C:/Users/zhile/Desktop/Nemo_nest/NeMo/examples/asr/conf/ssl/nest --config-name nest_fast-conformer model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 +trainer.limit_train_batches=5 +trainer.num_sanity_val_steps=0
```

### nest_ssl_project Training with Comparator

```powershell
# Use + prefix for keys not in config file
python tools/nest_training_with_comparator.py --config-path config --config-name nest_fast-conformer model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 +trainer.limit_train_batches=5 +trainer.num_sanity_val_steps=0
```

## Key Points

- Use `+key=value` to add new keys not in config file
- Use `key=value` to override existing keys
- Config file defaults (output_dir, seed, etc.) are already set, no need to specify

