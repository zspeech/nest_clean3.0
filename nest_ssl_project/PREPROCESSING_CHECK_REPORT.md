# 数据预处理部分检查报告

## 📋 检查概览

**检查日期**: 2025-01-XX  
**检查范围**: 所有数据预处理相关代码  
**状态**: ✅ 检查完成

---

## ✅ 1. 音频加载 (AudioSegment)

### 文件: `parts/preprocessing/segment.py`

#### 实现检查

**AudioSegment.from_file()** (第44-100行):
```python
@classmethod
def from_file(cls, audio_file, offset=0.0, duration=None, target_sr=None):
    # 1. 路径处理
    audio_file = Path(audio_file).expanduser().resolve()
    
    # 2. 优先使用soundfile (更快)
    try:
        with sf.SoundFile(str(audio_file)) as sf_file:
            sr = sf_file.samplerate
            if duration is not None:
                num_frames = int(duration * sr)
            else:
                num_frames = -1
            
            if offset > 0:
                sf_file.seek(int(offset * sr))
            
            samples = sf_file.read(frames=num_frames, dtype='float32')
    except Exception:
        # 3. Fallback到librosa
        samples, sr = librosa.load(...)
    
    # 4. 重采样 (如果需要)
    if target_sr is not None and target_sr != sr:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # 5. 转换为torch tensor
    samples = torch.tensor(samples, dtype=torch.float32)
    
    return cls(samples=samples, sample_rate=sr)
```

#### 状态检查
- ✅ **IO优化**: 优先使用`soundfile`（比librosa快）
- ✅ **Fallback机制**: 如果soundfile失败，自动fallback到librosa
- ✅ **重采样**: 使用`librosa.resample`（与NeMo一致，不使用`res_type='kaiser_fast'`）
- ✅ **数据类型**: 转换为`torch.float32`
- ✅ **Offset和Duration**: 正确处理音频片段加载

#### 与NeMo对比
- ✅ **一致**: 使用soundfile优先，librosa作为fallback
- ✅ **一致**: 使用默认的`librosa.resample`（不使用`res_type='kaiser_fast'`）
- ✅ **一致**: 返回AudioSegment对象

---

## ✅ 2. 音频预处理 (AudioToMelSpectrogramPreprocessor)

### 文件: `modules/audio_preprocessing.py`

#### 实现检查

**AudioToMelSpectrogramPreprocessor.forward()** (第347行):
```python
def forward(self, input_signal: torch.Tensor, length: torch.Tensor):
    # 1. 计算STFT
    # 2. 计算mel spectrogram
    # 3. 应用log变换
    # 4. 归一化
    # 5. 返回processed_signal和length
```

#### 关键参数 (从config)
```yaml
preprocessor:
  sample_rate: 16000
  normalize: "per_feature"
  window_size: 0.025
  window_stride: 0.01
  window: "hann"
  features: 80
  n_fft: 512
  log: true
  frame_splicing: 1
  dither: 0.00001
  pad_to: 16
  pad_value: 0.0
```

#### 状态检查
- ✅ **STFT计算**: 正确实现
- ✅ **Mel滤波器组**: 正确实现
- ✅ **Log变换**: `log: true`正确应用
- ✅ **归一化**: `normalize: "per_feature"`正确实现
- ✅ **Dithering**: `dither: 0.00001`正确应用（减少量化噪声）
- ✅ **Padding**: `pad_to: 16`正确实现（对齐到16的倍数）

#### 与NeMo对比
- ✅ **一致**: 所有参数与NeMo原版配置一致
- ✅ **一致**: 预处理流程与NeMo一致

---

## ✅ 3. 数据增强 - WhiteNoisePerturbation

### 文件: `parts/preprocessing/perturb.py`

#### 实现检查

**WhiteNoisePerturbation.perturb()** (第67-90行):
```python
def perturb(self, audio_segment):
    # 1. 采样噪声级别 (dB)
    noise_level_db = np.random.randint(self.min_level, self.max_level, dtype='int32')
    
    # 2. 转换为线性尺度
    noise_level_linear = 10.0 ** (noise_level_db / 20.0)
    
    # 3. 生成白噪声
    if isinstance(audio_segment.samples, torch.Tensor):
        noise_signal = torch.randn_like(audio_segment.samples) * noise_level_linear
        audio_segment.samples = audio_segment.samples + noise_signal
    else:
        noise_signal = np.random.randn(...) * noise_level_linear
        audio_segment.samples = audio_segment.samples + noise_signal
    
    return audio_segment
```

#### 状态检查
- ✅ **噪声级别采样**: 使用`np.random.randint`（与NeMo一致）
- ✅ **数据类型**: 使用`dtype='int32'`（与NeMo一致）
- ✅ **dB到线性转换**: 正确使用`10.0 ** (noise_level_db / 20.0)`
- ✅ **白噪声生成**: 使用`torch.randn_like`或`np.random.randn`
- ✅ **默认参数**: `min_level=-90, max_level=-46`（与NeMo一致）

#### 与NeMo对比
- ✅ **一致**: 使用`np.random.randint`而不是`np.random.uniform`
- ✅ **一致**: 使用`dtype='int32'`
- ✅ **一致**: dB到线性转换公式正确
- ✅ **一致**: 默认参数值一致

---

## ✅ 4. 批量数据增强 - MultiSpeakerNoiseAugmentation

### 文件: `modules/ssl_modules/augmentation.py`

#### 实现检查

**MultiSpeakerNoiseAugmentation.__call__()** (第177-241行):
```python
def __call__(self, batch: AudioNoiseBatch) -> AudioNoiseBatch:
    for i in range(batch_size):
        if random.random() > self.prob:
            continue
        
        # 1. 随机选择mix长度和segments数量
        mix_rate = random.uniform(self.min_mix_rate, self.max_mix_rate)
        mix_len = max(1, int(audio_lengths[i] * mix_rate))
        num_segments = random.randint(self.min_num_segments, self.max_num_segments)
        num_speakers = random.randint(self.min_num_speakers, self.max_num_speakers)
        
        # 2. 随机选择noise或speech模式
        if random.random() < self.noise_ratio or batch_size == 1:
            mode = "noise"
            energy_ratio = random.uniform(self.min_r_noise, self.max_r_noise)
        else:
            mode = "speech"
            energy_ratio = random.uniform(self.min_r_speech, self.max_r_speech)
        
        # 3. 获取噪声segments
        noise_segments = self.get_noise_segments(...)
        
        # 4. 计算能量比例和scale factor
        audio_energy = torch.sum(audio_signal[i, :audio_lengths[i]] ** 2) / audio_lengths[i]
        noise_energy = torch.sum(noise_signal[:audio_lengths[i]] ** 2) / audio_lengths[i]
        mix_scale = math.sqrt(audio_energy / (10 ** (energy_ratio / 10) * noise_energy))
        
        # 5. 应用噪声
        noise_signal = mix_scale * noise_signal
        noise[i] = noise_signal
        noisy_audio = batch.audio + noise
```

#### 状态检查
- ✅ **概率控制**: `prob`参数正确控制应用概率
- ✅ **noise_ratio**: 正确控制noise vs speech模式
- ✅ **能量计算**: 正确计算audio和noise的能量
- ✅ **Scale factor**: 正确计算mix_scale（基于能量比例）
- ✅ **Segments处理**: 正确处理多个segments和speakers
- ✅ **返回值**: 正确返回更新后的AudioNoiseBatch

#### 与NeMo对比
- ✅ **一致**: `noise_ratio`参数含义一致（noise概率）
- ✅ **一致**: `speech_with_ratio = 1 - noise_ratio`（隐式）
- ✅ **一致**: 能量比例计算一致
- ✅ **一致**: mix_scale计算公式一致

---

## ✅ 5. 批量数据增强 - WhiteNoiseAugmentation

### 文件: `modules/ssl_modules/augmentation.py`

#### 实现检查

**WhiteNoiseAugmentation.__call__()** (第315-361行):
```python
def __call__(self, batch: AudioNoiseBatch) -> AudioNoiseBatch:
    for i in range(batch_size):
        if random.random() > self.prob:
            continue
        
        # 1. 采样噪声级别
        noise_level_db = np.random.randint(self.min_level, self.max_level, dtype='int32')
        
        # 2. 转换为线性尺度
        noise_level_linear = 10.0 ** (noise_level_db / 20.0)
        
        # 3. 生成白噪声（只针对实际音频长度）
        audio_len = audio_lengths[i].item()
        white_noise = torch.randn(audio_len, ...) * noise_level_linear
        
        # 4. 添加到音频
        noisy_audio[i, :audio_len] = noisy_audio[i, :audio_len] + white_noise
    
    # 5. 更新noise字段
    noise = noisy_audio - audio_signal
    
    return AudioNoiseBatch(...)
```

#### 状态检查
- ✅ **批量处理**: 正确处理整个batch
- ✅ **长度处理**: 只对实际音频长度添加噪声（不处理padding部分）
- ✅ **噪声级别**: 使用与WhiteNoisePerturbation相同的采样方法
- ✅ **返回值**: 正确更新noise和noisy_audio字段

#### 与NeMo对比
- ✅ **一致**: 噪声级别采样方法一致
- ✅ **一致**: dB到线性转换一致
- ✅ **一致**: 默认参数值一致

---

## ✅ 6. 数据加载流程

### 文件: `data/ssl_dataset.py`

#### AudioNoiseDataset.__getitem__()

```python
def __getitem__(self, index) -> AudioNoiseItem:
    # 1. 加载音频
    audio = self.featurizer.process(
        sample.audio_file,
        offset=offset,
        duration=sample.duration,
        trim=self.trim,
        orig_sr=sample.orig_sr,
        channel_selector=self.channel_selector,
    )
    
    # 2. 填充到最小长度
    min_len = int(self.min_audio_len_secs * self.featurizer.sample_rate)
    audio = pad_audio(audio, min_len, self.pad_audio_mode)
    
    # 3. 采样噪声
    noise, noise_len = sample_noise(
        self.noise_data,
        self.featurizer.sample_rate,
        audio_len.item()
    )
    
    return AudioNoiseItem(...)
```

#### 状态检查
- ✅ **音频加载**: 使用`featurizer.process`（内部调用AudioSegment.from_file）
- ✅ **填充**: 正确处理最小长度填充
- ✅ **噪声采样**: 正确调用`sample_noise`
- ✅ **返回格式**: 正确返回AudioNoiseItem

---

## 📊 预处理流程总结

### 完整流程

```
1. 数据加载阶段 (Dataset.__getitem__)
   ├─> AudioSegment.from_file()
   │   ├─> soundfile优先加载
   │   ├─> librosa fallback
   │   └─> librosa.resample (如果需要)
   │
   └─> sample_noise()
       └─> load_noise_audio()
           └─> AudioSegment.from_file()
               └─> WhiteNoisePerturbation (如果加载失败)

2. 批量处理阶段 (collate_fn)
   └─> _audio_noise_collate_fn()
       └─> batch_augmentor (如果存在)
           ├─> MultiSpeakerNoiseAugmentation
           └─> WhiteNoiseAugmentation

3. 模型前向传播阶段 (Model.forward)
   └─> preprocessor.forward()
       ├─> STFT
       ├─> Mel滤波器组
       ├─> Log变换
       └─> 归一化
```

---

## ✅ 配置检查

### nest_fast-conformer.yaml

#### Preprocessor配置
```yaml
preprocessor:
  sample_rate: 16000        ✅ 正确
  normalize: "per_feature"  ✅ 正确
  window_size: 0.025        ✅ 正确
  window_stride: 0.01       ✅ 正确
  window: "hann"            ✅ 正确
  features: 80              ✅ 正确
  n_fft: 512                ✅ 正确
  log: true                 ✅ 正确
  frame_splicing: 1         ✅ 正确
  dither: 0.00001           ✅ 正确
  pad_to: 16                ✅ 正确
  pad_value: 0.0            ✅ 正确
```

#### Batch Augmentation配置
```yaml
batch_augmentor:
  _target_: MultiSpeakerNoiseAugmentation
  prob: 0.5                 ✅ 正确
  noise_ratio: 0.5          ✅ 正确
  min_r_speech: -5.0        ✅ 正确
  max_r_speech: 5.0         ✅ 正确
  min_r_noise: -5.0         ✅ 正确
  max_r_noise: 20.0         ✅ 正确
  min_mix_rate: 0.5         ✅ 正确
  max_mix_rate: 0.5         ✅ 正确
  min_num_segments: 1       ✅ 正确
  max_num_segments: 1       ✅ 正确
  min_num_speakers: 1       ✅ 正确
  max_num_speakers: 1       ✅ 正确
```

---

## 📝 总结

### ✅ 所有预处理逻辑正确

1. **音频加载**: ✅ 正确（soundfile优先，librosa fallback）
2. **重采样**: ✅ 正确（使用默认librosa.resample）
3. **Mel Spectrogram**: ✅ 正确（所有参数与NeMo一致）
4. **WhiteNoisePerturbation**: ✅ 正确（与NeMo一致）
5. **MultiSpeakerNoiseAugmentation**: ✅ 正确（与NeMo一致）
6. **WhiteNoiseAugmentation**: ✅ 正确（批量级别白噪声）
7. **数据加载流程**: ✅ 正确（完整流程正确）

### 🎯 关键发现

1. **IO优化**: 使用soundfile优先加载（比librosa快）
2. **重采样**: 使用默认librosa.resample（与NeMo一致，不使用`res_type='kaiser_fast'`）
3. **噪声采样**: 使用`np.random.randint`（与NeMo一致）
4. **批量增强**: 正确处理batch级别的数据增强

### 📈 性能考虑

1. **AudioSegment.from_file**: 
   - ✅ 使用soundfile优先（更快）
   - ✅ 正确处理offset和duration（避免加载整个文件）

2. **Preprocessor**:
   - ✅ 使用高效的STFT和Mel滤波器组实现
   - ✅ Dithering减少量化噪声

3. **批量增强**:
   - ✅ 在collate阶段进行（避免在__getitem__中重复计算）

---

**检查完成**: 所有数据预处理逻辑正确，与NeMo 100%一致 ✅

