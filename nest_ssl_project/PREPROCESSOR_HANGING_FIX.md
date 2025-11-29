# Preprocessor 卡住问题修复指南

## 🔍 问题描述

Rank 3 卡在了 preprocess（预处理）阶段，导致其他 rank 在等待。

## 📊 已添加的调试信息

### 1. **Preprocessor 调用前后调试**

在两个 preprocessor 调用处添加了详细的调试信息：

**input_signal 预处理**：
```python
print(f"[Rank {self.global_rank}] About to call preprocessor for input_signal, ...")
processed_signal, processed_signal_length = self.preprocessor(...)
print(f"[Rank {self.global_rank}] Preprocessor for input_signal completed, ...")
```

**noisy_input_signal 预处理**：
```python
print(f"[Rank {self.global_rank}] About to call preprocessor for noisy_input_signal, ...")
processed_noisy_input_signal, processed_noisy_input_signal_length = self.preprocessor(...)
print(f"[Rank {self.global_rank}] Preprocessor for noisy_input_signal completed, ...")
```

## 🔍 可能的原因

### 1. **数据加载问题**
- Rank 3 的数据文件可能损坏或无法访问
- 文件 I/O 阻塞
- 网络文件系统延迟

### 2. **Preprocessor 内部问题**
- Preprocessor 的某些操作（如 FFT、mel 变换）可能卡住
- GPU 内存不足
- CUDA 操作未完成

### 3. **数据形状问题**
- 输入数据形状不一致
- 输入数据包含 NaN 或 Inf
- 输入数据长度异常

### 4. **DDP 同步问题**
- 如果某个 rank 在预处理时卡住，其他 rank 会在后续的 DDP 同步点等待

## 🛠️ 调试步骤

### 步骤 1: 查看预处理输出

运行训练时，查看以下输出：

```
[Rank 0] About to call preprocessor for input_signal, input_signal.shape=...
[Rank 1] About to call preprocessor for input_signal, input_signal.shape=...
[Rank 2] About to call preprocessor for input_signal, input_signal.shape=...
[Rank 3] About to call preprocessor for input_signal, input_signal.shape=...
[Rank 0] Preprocessor for input_signal completed, processed_signal.shape=...
[Rank 1] Preprocessor for input_signal completed, processed_signal.shape=...
[Rank 2] Preprocessor for input_signal completed, processed_signal.shape=...
[Rank 3] ... (卡住，没有输出)
```

### 步骤 2: 定位卡住位置

根据输出判断：

- **如果看到 "About to call preprocessor" 但没有 "Preprocessor completed"**：
  - 问题在 preprocessor 内部
  - 检查输入数据是否有问题
  - 检查 preprocessor 的配置

- **如果某个 rank 没有输出 "About to call preprocessor"**：
  - 问题在数据加载阶段
  - 检查 `__getitem__` 的输出
  - 检查数据文件是否可访问

### 步骤 3: 检查数据加载

查看 `__getitem__` 的调试输出：
```
[Rank 3] __getitem__(568) called
[Rank 3] Loading audio from ..., index=568
[Rank 3] Audio loaded, shape=..., index=568
```

如果看到 "Loading audio" 但没有 "Audio loaded"，说明数据加载卡住。

### 步骤 4: 检查特定样本

如果 rank 3 总是卡在同一个 batch，检查：
- 该 batch 对应的数据文件
- 文件是否损坏
- 文件路径是否正确

## 🔧 可能的解决方案

### 方案 1: 检查数据文件

检查 rank 3 的数据文件：
```python
# 在数据加载时添加验证
if self.global_rank == 3:
    print(f"[Rank 3] Checking file: {sample.audio_file}", flush=True)
    if not os.path.exists(sample.audio_file):
        print(f"[Rank 3] ERROR: File not found: {sample.audio_file}", flush=True)
```

### 方案 2: 添加超时机制

在 preprocessor 调用处添加超时（仅用于调试）：
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Preprocessor call timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 秒超时
try:
    processed_signal, processed_signal_length = self.preprocessor(...)
finally:
    signal.alarm(0)
```

### 方案 3: 检查 GPU 内存

使用 `nvidia-smi` 检查 GPU 内存：
```bash
watch -n 1 nvidia-smi
```

如果 rank 3 的 GPU 内存使用异常，可能是 OOM 导致卡住。

### 方案 4: 添加错误处理

在 preprocessor 调用处添加 try-catch：
```python
try:
    processed_signal, processed_signal_length = self.preprocessor(...)
except Exception as e:
    print(f"[Rank {self.global_rank}] ERROR in preprocessor: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise
```

### 方案 5: 检查输入数据

在 preprocessor 调用前验证输入：
```python
# 检查 NaN/Inf
if torch.isnan(input_signal).any() or torch.isinf(input_signal).any():
    print(f"[Rank {self.global_rank}] WARNING: Input contains NaN or Inf!", flush=True)

# 检查形状
if input_signal.dim() != 2:
    print(f"[Rank {self.global_rank}] ERROR: Invalid input shape: {input_signal.shape}", flush=True)
```

## 📝 下一步

1. **运行训练并查看调试输出**
2. **根据输出定位 rank 3 卡住的具体位置**
3. **检查 rank 3 的数据文件**
4. **如果问题在 preprocessor 内部，检查输入数据和配置**

## 🔗 相关文档

- [ENCODER_INPUT_HANGING_DEBUG.md](ENCODER_INPUT_HANGING_DEBUG.md) - Encoder 输入卡住调试
- [BATCH_71_HANGING_DEBUG.md](BATCH_71_HANGING_DEBUG.md) - Batch 71 卡住调试
- [DDP_TROUBLESHOOTING.md](DDP_TROUBLESHOOTING.md) - DDP 故障排除

---

**更新日期**: 2025-01-XX  
**版本**: 1.0  
**状态**: 🔴 调试中


