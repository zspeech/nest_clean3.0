# 数据加载性能优化

## 已实施的优化

### 1. ✅ 优化噪声采样重试次数
- **之前：** `max_trial: 20` 在 `sample_noise` 中
- **现在：** `max_trial: 5`（减少75%的重试次数）
- **预期提升：** +20-30% 数据加载速度

### 2. ✅ 优化噪声加载重试次数
- **之前：** `max_trial: 100` 在 `load_noise_audio` 中
- **现在：** 限制为最多10次（减少90%的重试次数）
- **预期提升：** +10-20% 数据加载速度

### 3. ✅ 优化音频重采样
- **之前：** 使用默认的 `librosa.resample`
- **现在：** 使用 `res_type='kaiser_fast'`（更快的重采样算法）
- **预期提升：** +10-15% 数据加载速度

### 4. ✅ 优化音频加载逻辑
- 优先使用 `soundfile`（比 `librosa` 快）
- 在 `soundfile` 加载时直接处理重采样，避免fallback到librosa

## 主要性能瓶颈

### 1. ⚠️ 单进程数据加载（Windows）
**问题：** `num_workers: 0` 导致所有数据加载在主进程中进行
**影响：** 这是最大的瓶颈，GPU会等待CPU加载数据

**解决方案：**
- **Windows：** 增加 `batch_size` 来补偿（例如8-16）
- **Linux/Mac：** 使用 `num_workers: 4-8`

### 2. ⚠️ 每次都要从磁盘读取文件
**问题：** 没有缓存机制，每次都要读取音频文件
**影响：** I/O成为瓶颈，特别是使用HDD时

**解决方案：**
- 使用SSD存储数据集
- 使用tarred数据集（预处理的音频，减少I/O）
- 增加 `batch_size` 减少相对I/O开销

### 3. ⚠️ 每次都要加载噪声音频
**问题：** 每个样本都要从噪声manifest中随机加载噪声
**影响：** 额外的I/O开销

**解决方案：**
- 如果 `noise_manifest: null`，噪声会在batch augmentation中生成，更快
- 或者使用预处理的噪声数据

## 进一步优化建议

### 立即可用（Windows）

1. **增加 batch_size**
   ```yaml
   batch_size: 8  # 或16
   ```
   **预期提升：** +50-100% 吞吐量

2. **使用SSD存储**
   - 将数据集放在SSD上
   - **预期提升：** +30-50% 数据加载速度

3. **减少验证频率**
   ```yaml
   val_check_interval: 0.5  # 每半个epoch验证一次
   ```

### 高级优化

1. **使用 Tarred 数据集**
   ```yaml
   is_tarred: true
   tarred_audio_filepaths: ["path/to/tarred/files"]
   ```
   **预期提升：** +50-100% 数据加载速度

2. **预计算特征（如果可能）**
   - 预处理音频为mel spectrogram
   - 保存为numpy文件
   - 直接加载特征而不是原始音频

3. **使用 WSL2（Windows）**
   - 允许使用 `num_workers > 0`
   - **预期提升：** +200-400% 数据加载速度

## 性能监控

### 检查数据加载时间

在训练脚本中添加：

```python
import time

# 在训练循环中
start_time = time.time()
batch = next(iter(train_dataloader))
load_time = time.time() - start_time
print(f"Data loading time: {load_time:.3f}s")
```

**目标：** 数据加载时间应该 < 训练时间

### 检查GPU利用率

```bash
# Windows
nvidia-smi -l 1

# Linux/Mac  
watch -n 1 nvidia-smi
```

**如果GPU利用率 < 50%：** 数据加载是瓶颈
- 增加 `batch_size`
- 使用 `num_workers > 0`（Linux/Mac）
- 使用tarred数据集

## 预期性能提升

| 优化项 | 预期提升 |
|--------|---------|
| 减少重试次数 | +20-30% |
| 优化重采样 | +10-15% |
| batch_size: 2 → 8 | +50-100% |
| 使用SSD | +30-50% |
| 使用tarred数据集 | +50-100% |
| num_workers: 0 → 4 (Linux) | +200-400% |

**组合优化预期：** 在Windows上，组合所有优化可以获得 **2-3倍** 的性能提升。

## 故障排除

### 如果数据加载仍然很慢：

1. **检查存储速度**
   ```bash
   # Windows: 检查磁盘速度
   # 使用CrystalDiskMark等工具
   ```

2. **检查音频文件格式**
   - WAV格式通常比FLAC/MP3快
   - 如果可能，转换为WAV格式

3. **减少音频长度**
   ```yaml
   max_duration: 30.0  # 从60.0减少到30.0
   ```

4. **禁用不必要的augmentation**
   ```yaml
   batch_augmentor:
     prob: 0.0  # 临时禁用以测试
   ```

