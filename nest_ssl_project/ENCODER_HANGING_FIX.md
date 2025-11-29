# Encoder 卡住问题修复指南

## 🔍 问题描述

训练在 encoder 调用处卡住，可能的原因包括：
1. DDP 同步死锁（某个 rank 卡住导致所有 rank 等待）
2. 输入数据问题（NaN/Inf 值、无效形状）
3. CUDA 操作未同步完成
4. BatchNorm 同步问题

## ✅ 已添加的修复

### 1. **输入验证和调试信息**
在 encoder 调用前添加了详细的输入检查：
- 检查输入张量的形状、设备、数据类型
- 检查 NaN 和 Inf 值
- 打印详细的调试信息

### 2. **CUDA 同步点**
在 encoder 调用前后添加了 `torch.cuda.synchronize()`：
- 确保所有之前的 CUDA 操作完成
- 帮助定位卡住的具体位置

### 3. **错误处理**
添加了 try-catch 块：
- 捕获 encoder 调用中的异常
- 打印详细的错误信息和堆栈跟踪
- 帮助快速定位问题

## 📊 调试步骤

### 步骤 1: 查看调试输出

运行训练时，查看以下输出：

```
[Rank X] Forward: Calling encoder...
[Rank X] Forward: Encoder input - audio_signal.shape=..., device=..., has_nan=..., has_inf=...
[Rank X] Forward: CUDA synchronized before encoder
[Rank X] Forward: Encoder completed, encoded.shape=...
```

### 步骤 2: 定位卡住位置

根据输出判断：

- **如果看到 "CUDA synchronized before encoder" 但没有 "Encoder completed"**：
  - 问题在 encoder 内部
  - 检查是否有特定输入导致 encoder 卡住
  - 检查 encoder 内部是否有死锁

- **如果看到 "has_nan=True" 或 "has_inf=True"**：
  - 输入数据包含 NaN 或 Inf 值
  - 检查预处理步骤
  - 检查数据加载过程

- **如果某个 rank 没有输出**：
  - 该 rank 可能在 encoder 之前就卡住了
  - 检查数据加载是否正常
  - 检查 DDP 初始化

### 步骤 3: 检查 DDP 同步

如果怀疑是 DDP 同步问题：

1. **检查所有 rank 的输出**：
   - 确保所有 rank 都到达 encoder 调用
   - 如果某个 rank 没有输出，说明它在更早的地方卡住了

2. **检查 GPU 利用率**：
   ```bash
   nvidia-smi
   ```
   - 如果某个 GPU 利用率一直为 0%，说明该 rank 卡住了

3. **检查内存使用**：
   - 如果某个 GPU 内存使用异常，可能是 OOM 导致卡住

## 🛠️ 可能的解决方案

### 方案 1: 检查输入数据

如果输入包含 NaN/Inf：
```python
# 在预处理后添加检查
if torch.isnan(processed_signal).any():
    print(f"Warning: NaN detected in processed_signal")
    processed_signal = torch.nan_to_num(processed_signal, nan=0.0)
```

### 方案 2: 禁用 persistent_workers

如果怀疑是 DataLoader worker 问题：
```yaml
train_ds:
  persistent_workers: false
  num_workers: 0  # 临时禁用多进程加载
```

### 方案 3: 检查 DDP 配置

确保 DDP 配置正确：
```yaml
trainer:
  strategy: auto  # 或 ddp
  sync_batchnorm: true  # 重要！
```

### 方案 4: 添加超时机制（高级）

如果需要，可以添加超时机制来检测卡住：
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Encoder call timed out")

# 在 encoder 调用前设置超时（仅用于调试）
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 60 秒超时
try:
    encoded, encoded_len = self.encoder(...)
finally:
    signal.alarm(0)  # 取消超时
```

## 📝 下一步

1. **运行训练并查看调试输出**
2. **根据输出定位卡住的具体位置**
3. **检查输入数据是否有问题**
4. **如果问题持续，尝试禁用某些优化（如 persistent_workers）**

## 🔗 相关文档

- [BATCH_71_HANGING_DEBUG.md](BATCH_71_HANGING_DEBUG.md) - Batch 71 卡住问题调试
- [DDP_TROUBLESHOOTING.md](DDP_TROUBLESHOOTING.md) - DDP 故障排除
- [DDP_DEBUG_GUIDE.md](DDP_DEBUG_GUIDE.md) - DDP 调试指南

---

**更新日期**: 2025-01-XX  
**版本**: 1.0


