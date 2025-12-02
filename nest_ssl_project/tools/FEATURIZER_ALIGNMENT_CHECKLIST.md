# FilterbankFeatures 跨环境对齐检查清单

## 随机性来源

### 1. Dither（抖动噪声）
- **位置**: `forward()` 中的 `torch.randn_like(x)`
- **影响**: 训练时添加随机噪声
- **解决方案**: 
  - 配置中设置 `dither: 0.0`
  - 或者在对齐测试时设置 `model.training = False`

### 2. nb_augmentation（窄带增强）
- **位置**: NeMo 原版有 `nb_augmentation_prob` 参数
- **影响**: 随机遮蔽高频部分
- **解决方案**: 确保 `nb_augmentation_prob: 0.0`（默认值）

## 跨环境差异来源

### 1. librosa.filters.mel
- **问题**: 不同 librosa 版本的 mel filterbank 计算可能有微小差异
- **检查命令**:
  ```python
  import librosa
  print(librosa.__version__)
  ```
- **解决方案**: 确保两个环境的 librosa 版本完全一致

### 2. torch.stft
- **问题**: 不同 PyTorch 版本的 FFT 实现可能有差异
- **检查命令**:
  ```python
  import torch
  print(torch.__version__)
  ```
- **解决方案**: 确保两个环境的 PyTorch 版本完全一致

### 3. torch.floor_divide
- **问题**: 在某些 PyTorch 版本中，`floor_divide` 对浮点数的处理可能不同
- **影响**: `get_seq_len` 计算的输出长度可能差 1
- **解决方案**: 
  ```python
  # 更明确的整数除法
  seq_len = ((seq_len + pad_amount - self.n_fft) // self.hop_length).to(torch.long)
  ```

### 4. 浮点精度差异
- **位置**: 
  - `normalize_batch` 中的均值/标准差计算
  - `torch.log` 计算
  - `torch.sqrt` 计算
- **影响**: CPU vs GPU、float32 vs float16 可能有微小差异
- **解决方案**: 
  - 使用 `torch.amp.autocast(enabled=False)` 禁用 autocast
  - 确保两个环境使用相同的设备类型（都用 CPU 或都用 GPU）

### 5. Window 函数
- **位置**: `torch.hann_window` 等
- **影响**: 不同 PyTorch 版本的窗函数实现可能有微小差异
- **解决方案**: 确保 PyTorch 版本一致

## 对齐测试步骤

1. **检查库版本**:
   ```bash
   pip freeze | grep -E "torch|librosa|numpy"
   ```

2. **禁用所有随机性**:
   ```yaml
   model:
     preprocessor:
       dither: 0.0
     train_ds:
       batch_augmentor:
         prob: 0.0
   ```

3. **使用相同设备**:
   ```yaml
   trainer:
     accelerator: cpu  # 或者都用 gpu
   ```

4. **设置随机种子**:
   ```python
   import torch
   import numpy as np
   import random
   
   seed = 42
   torch.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)
   ```

5. **比较 mel filterbank**:
   ```python
   # 在两个环境中分别运行
   import librosa
   import torch
   
   fb = librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000, norm='slaney')
   print(f"Mel filterbank shape: {fb.shape}")
   print(f"First row sum: {fb[0].sum()}")
   print(f"First 5 values: {fb[0, :5]}")
   ```

## 已知的 PyTorch 版本差异

- PyTorch 1.x vs 2.x: `torch.floor_divide` 行为变化
- PyTorch < 1.8: `torch.stft` 返回格式不同
- PyTorch 2.0+: 默认使用新的 FFT 实现

## 推荐的环境配置

为确保跨环境一致性，建议使用以下版本：
```
torch>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
```

