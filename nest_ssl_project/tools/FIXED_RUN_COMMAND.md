# Fixed Run Command - NeMo Training with Output Saver

## Problem
The config path `../conf/ssl/nest` is relative to NeMo's `examples/asr` directory, not `nest_ssl_project` directory.

## Solution

### Option 1: Run from NeMo Root Directory (Recommended)

```powershell
# Navigate to NeMo root directory first
cd C:\Users\zhile\Desktop\Nemo_nest\NeMo

# Then run (using NeMo's original script with modifications, or copy nemo_training_with_saver.py to NeMo/tools/)
python examples/asr/speech_pretraining/masked_token_pred_pretrain.py output_dir=./saved_nemo_outputs seed=42 save_steps="0,1,2,3,4" model.train_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

### Option 2: Use Absolute Config Path

```powershell
# From nest_ssl_project directory
python tools/nemo_training_with_saver.py --config-path C:/Users/zhile/Desktop/Nemo_nest/NeMo/examples/asr/conf/ssl/nest --config-name nest_fast-conformer output_dir=./saved_nemo_outputs seed=42 save_steps="0,1,2,3,4" model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

### Option 3: Copy Script to NeMo and Run from There

```powershell
# Copy the script to NeMo
copy nest_ssl_project\tools\nemo_training_with_saver.py NeMo\tools\
copy nest_ssl_project\tools\training_output_saver.py NeMo\tools\

# Navigate to NeMo root
cd C:\Users\zhile\Desktop\Nemo_nest\NeMo

# Run from NeMo root
python tools/nemo_training_with_saver.py --config-path examples/asr/conf/ssl/nest --config-name nest_fast-conformer output_dir=./saved_nemo_outputs seed=42 save_steps="0,1,2,3,4" model.train_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=../nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

## Recommended: Use NeMo's Original Script with Callback

The easiest way is to modify NeMo's original training script to add the callback:

1. Copy `training_output_saver.py` to NeMo environment
2. Modify `NeMo/examples/asr/speech_pretraining/masked_token_pred_pretrain.py` to add the callback
3. Run from NeMo root directory

