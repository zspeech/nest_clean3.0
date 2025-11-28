# Dummy SSL Dataset for Debugging

This directory contains a small dummy dataset for debugging the SSL training pipeline.

## Dataset Structure

```
dummy_ssl/
├── train/
│   ├── train_0000.wav
│   ├── train_0001.wav
│   └── ... (10 files total)
├── val/
│   ├── val_0000.wav
│   └── val_0001.wav (2 files total)
├── train_manifest.json
└── val_manifest.json
```

## Dataset Details

- **Training files**: 10 audio files, 3 seconds each
- **Validation files**: 2 audio files, 3 seconds each
- **Sample rate**: 16000 Hz
- **Format**: WAV files with random noise (dummy data)

## Usage

The config file (`config/nest_fast-conformer.yaml`) has been pre-configured to use this dataset:

- `model.train_ds.manifest_filepath`: `data/dummy_ssl/train_manifest.json`
- `model.validation_ds.manifest_filepath`: `data/dummy_ssl/val_manifest.json`
- `model.train_ds.batch_size`: 2 (small batch for debugging)
- `trainer.devices`: 1 (single GPU for debugging)
- `trainer.max_steps`: 100 (small number for quick debugging)

## Regenerating the Dataset

To regenerate the dataset with different parameters:

```bash
python tools/prepare_dummy_ssl_data.py \
    --num-train 10 \
    --num-val 2 \
    --duration 3.0 \
    --sample-rate 16000
```

Options:
- `--num-train`: Number of training files (default: 10)
- `--num-val`: Number of validation files (default: 2)
- `--duration`: Duration of each audio file in seconds (default: 3.0)
- `--sample-rate`: Sample rate for audio files (default: 16000)
- `--data-dir`: Custom data directory (default: `data/dummy_ssl`)

## Running Training

To start training with this dataset:

```bash
cd nest_ssl_project
python train.py
```

Or override config values:

```bash
python train.py \
    model.train_ds.batch_size=4 \
    trainer.devices=1 \
    trainer.max_steps=50
```

## Notes

- This is a **dummy dataset** with random noise, suitable only for debugging the training pipeline
- For actual training, replace with real speech data
- The manifest files use relative paths, so they should work regardless of the absolute path to the project

