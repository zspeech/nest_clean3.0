# Ready-to-Run Commands

## NeMo Training with Output Saver

**Important**: Run this command from NeMo root directory or examples/asr directory.

### Windows PowerShell (Copy and Run - Single Line)

```powershell
# From NeMo root directory
python examples/asr/speech_pretraining/masked_token_pred_pretrain.py output_dir=./saved_nemo_outputs seed=42 save_steps="0,1,2,3,4" model.train_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0

# Or if you copied nemo_training_with_saver.py to NeMo/tools/
python tools/nemo_training_with_saver.py --config-path examples/asr/conf/ssl/nest --config-name nest_fast-conformer output_dir=./saved_nemo_outputs seed=42 save_steps="0,1,2,3,4" model.train_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

### Linux/Mac Bash (Copy and Run - Multi-line)

```bash
# From NeMo root directory
python examples/asr/speech_pretraining/masked_token_pred_pretrain.py \
    output_dir=./saved_nemo_outputs \
    seed=42 \
    save_steps="0,1,2,3,4" \
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
python examples/asr/speech_pretraining/masked_token_pred_pretrain.py output_dir=./saved_nemo_outputs seed=42 save_steps="0,1,2,3,4" model.train_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

## nest_ssl_project Training with Comparator

### Windows PowerShell (Copy and Run - Single Line)

```powershell
python tools/nest_training_with_comparator.py --config-path config --config-name nest_fast-conformer saved_outputs_dir=./saved_nemo_outputs comparison_output_dir=./comparison_results seed=42 atol=1e-5 rtol=1e-5 model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

### Linux/Mac Bash (Copy and Run - Multi-line)

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

### Windows CMD (Copy and Run - Single Line)

```cmd
python tools/nest_training_with_comparator.py --config-path config --config-name nest_fast-conformer saved_outputs_dir=./saved_nemo_outputs comparison_output_dir=./comparison_results seed=42 atol=1e-5 rtol=1e-5 model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

## Notes

- All paths use relative paths from project root
- Uses dummy data from `nest_ssl_project/data/dummy_ssl/`
- Saves outputs for steps 0,1,2,3,4
- Limits training to 5 batches for quick testing
- Both commands use same seed=42 for fair comparison

