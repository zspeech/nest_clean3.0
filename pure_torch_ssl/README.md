# Pure PyTorch SSL Model

This is a standalone implementation of the NEST (Noise-resilient Early-fusion Speech Tokenizer) model in pure PyTorch.

**No dependencies on:**
- PyTorch Lightning
- NeMo
- OmegaConf
- Hydra

**Only requires:**
- `torch` (PyTorch)
- `numpy`
- `pyyaml`
- `librosa` (for audio preprocessing)

## Installation

```bash
pip install torch numpy pyyaml librosa
```

## Usage

### 1. Create Model from Config File

```python
from ssl_model import PureTorchSSLModel

# Load from YAML config
model = PureTorchSSLModel.from_config_file("config.yaml")
model.to('cuda')
```

### 2. Create Model with Explicit Parameters

```python
from ssl_model import PureTorchSSLModel

model = PureTorchSSLModel(
    # Audio preprocessing
    sample_rate=16000,
    features=80,           # Mel features
    window_size=0.025,     # 25ms window
    window_stride=0.01,    # 10ms stride
    
    # Conformer encoder
    n_layers=17,
    d_model=512,
    n_heads=8,
    subsampling="dw_striding",
    subsampling_factor=8,
    
    # Quantizer
    num_classes=8192,
    num_books=1,
    code_dim=16,
    
    # Masking
    block_size=40,
    mask_prob=0.01,
)
model.to('cuda')
```

### 3. Training Loop

```python
from ssl_model import PureTorchSSLModel, create_optimizer, create_scheduler
import yaml

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Create model
model = PureTorchSSLModel.from_config_file("config.yaml")
model.to('cuda')
model.train()

# Create optimizer and scheduler
optimizer = create_optimizer(model, cfg)
scheduler = create_scheduler(optimizer, cfg)

# Training loop
for batch in dataloader:
    # batch should have: audio, audio_len, noisy_audio, noisy_audio_len
    optimizer.zero_grad()
    
    loss = model.training_step(batch)
    loss.backward()
    
    optimizer.step()
    if scheduler:
        scheduler.step()
    
    print(f"Loss: {loss.item():.4f}")
```

### 4. Forward Pass (Inference)

```python
import torch

model.eval()
with torch.no_grad():
    log_probs, encoded_len, masks, tokens = model.forward(
        input_signal=clean_audio,           # [B, T] clean audio
        input_signal_length=clean_len,      # [B] lengths
        noisy_input_signal=noisy_audio,     # [B, T] noisy audio
        noisy_input_signal_length=noisy_len,
        apply_mask=False,                   # No masking for inference
    )
```

### 5. Save and Load Checkpoints

```python
# Save
model.save_checkpoint("checkpoint.pt", optimizer=optimizer, epoch=10, step=1000)

# Load
model = PureTorchSSLModel.from_checkpoint("checkpoint.pt")

# Or load into existing model
info = model.load_checkpoint("checkpoint.pt", optimizer=optimizer)
print(f"Resuming from epoch {info['epoch']}, step {info['step']}")
```

## Batch Format

The model accepts batches in multiple formats:

### Named tuple / dataclass
```python
@dataclass
class AudioNoiseBatch:
    audio: torch.Tensor           # [B, T] clean audio
    audio_len: torch.Tensor       # [B] lengths
    noise: torch.Tensor           # [B, T] noise (optional)
    noise_len: torch.Tensor       # [B] lengths (optional)
    noisy_audio: torch.Tensor     # [B, T] noisy audio
    noisy_audio_len: torch.Tensor # [B] lengths
```

### Tuple
```python
batch = (audio, audio_len, noise, noise_len, noisy_audio, noisy_audio_len)
```

### Dict
```python
batch = {
    'audio': audio,
    'audio_len': audio_len,
    'noisy_audio': noisy_audio,
    'noisy_audio_len': noisy_audio_len,
}
```

## Config File Format

Example YAML config:

```yaml
model:
  sample_rate: 16000
  num_classes: 8192
  num_books: 1
  code_dim: 16
  squeeze_single: false
  mask_position: pre_conv
  
  preprocessor:
    features: 80
    window_size: 0.025
    window_stride: 0.01
    n_fft: 512
    normalize: per_feature
  
  encoder:
    n_layers: 17
    d_model: 512
    n_heads: 8
    subsampling: dw_striding
    subsampling_factor: 8
    subsampling_conv_channels: 256
    ff_expansion_factor: 4
    conv_kernel_size: 9
    dropout: 0.1
  
  masking:
    block_size: 40
    mask_prob: 0.01
  
  loss:
    mask_threshold: 0.8
  
  optim:
    name: adamw
    lr: 1e-4
    betas: [0.9, 0.999]
    weight_decay: 1e-3
    sched:
      name: noam
      d_model: 512
      warmup_steps: 25000
      min_lr: 1e-6
```

## File Structure

```
pure_torch_ssl/
├── README.md
├── ssl_model.py          # Main model class
├── core/                 # Core utilities
│   ├── classes/          # Base classes
│   └── neural_types/     # Type definitions
├── modules/
│   ├── audio_preprocessing.py   # Mel spectrogram
│   ├── conformer_encoder.py     # Conformer encoder
│   └── ssl_modules/
│       ├── masking.py           # Random block masking
│       ├── quantizers.py        # Vector quantizer
│       └── multi_softmax_decoder.py
└── losses/
    └── ssl_losses/
        └── mlm.py        # Masked LM loss
```

## Model Architecture

The NEST model consists of:

1. **Preprocessor**: Converts audio waveform to mel spectrogram
2. **Quantizer**: Generates discrete tokens from clean audio (for targets)
3. **Mask Processor**: Applies random block masking to noisy audio
4. **Conformer Encoder**: Encodes masked spectrogram
5. **Decoder**: Predicts quantized tokens from encoder output
6. **Loss**: Cross-entropy loss on masked positions

## Citation

If you use this code, please cite the NEST paper:
```
@article{nest2024,
  title={NEST: Noise-resilient Early-fusion Speech Tokenizer},
  author={...},
  journal={arXiv preprint arXiv:2408.13106},
  year={2024}
}
```

