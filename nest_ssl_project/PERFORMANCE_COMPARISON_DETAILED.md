# 详细性能对比和优化指南

## 与NeMo原版对比

### 1. DataLoader配置 ✅ 已对齐

**NeMo原版：**
```python
torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=config['batch_size'],
    collate_fn=collate_fn,
    drop_last=config.get('drop_last', False),
    shuffle=shuffle,
    num_workers=config.get('num_workers', 0),
    pin_memory=config.get('pin_memory', False),
)
```

**我们的实现：** ✅ 已对齐（移除了persistent_workers和prefetch_factor）

**说明：** NeMo不使用`persistent_workers`和`prefetch_factor`，因为它们在某些环境下可能导致问题。

### 2. Preprocessor调用 ⚠️ 设计限制

**NeMo原版：** `EncDecDenoiseMaskedTokenPredModel.forward()`中调用preprocessor两次：
1. `processed_signal` - 用于quantizer生成tokens
2. `processed_noisy_input_signal` - 用于encoder训练

**我们的实现：** ✅ 已对齐（与NeMo完全一致）

**性能影响：** 这是设计上的要求，无法避免。每个batch需要处理两个音频信号（原始音频和加噪音频）。

### 3. DDP配置 ✅ 已优化

**NeMo推荐配置：**
```yaml
trainer:
  strategy:
    _target_: lightning.pytorch.strategies.DDPStrategy
    gradient_as_bucket_view: true  # 关键优化
    static_graph: false
  sync_batchnorm: true
```

**我们的实现：** ✅ 已添加（见`nest_fast-conformer_ddp_example.yaml`）

### 4. 配置参数对比

| 参数 | NeMo默认 | 我们的默认 | 说明 |
|------|---------|-----------|------|
| `batch_size` | 16 | 2 | 我们的默认值较小（用于小数据集） |
| `num_workers` | 8 | 0 | Windows兼容性（Linux应设为8） |
| `pin_memory` | false (train), true (val) | true | 已对齐 |
| `persistent_workers` | 未使用 | 已移除 | 与NeMo对齐 |
| `prefetch_factor` | 未使用 | 已移除 | 与NeMo对齐 |

## 性能优化建议

### 1. 增加batch_size（最重要）

```yaml
model:
  train_ds:
    batch_size: 8  # 或16，根据GPU内存调整
```

**预期提升：** +50-100% 吞吐量

### 2. 使用多GPU训练（Linux）

```yaml
trainer:
  devices: -1  # 使用所有GPU
  strategy:
    _target_: lightning.pytorch.strategies.DDPStrategy
    gradient_as_bucket_view: true
  sync_batchnorm: true
```

**预期提升：** +N倍（N=GPU数量）

### 3. 使用混合精度训练

```yaml
trainer:
  precision: bf16-mixed  # 或16-mixed，如果GPU支持
```

**预期提升：** +50-100% 速度（如果GPU支持）

### 4. 增加num_workers（Linux/Mac）

```yaml
model:
  train_ds:
    num_workers: 8  # Linux/Mac可以使用，Windows必须为0
```

**预期提升：** +20-30% 数据加载速度

### 5. 使用SSD存储数据集

**预期提升：** +30-50% 数据加载速度

## 已知性能瓶颈

### 1. Preprocessor双重调用（无法避免）
- **原因：** 设计要求，需要处理原始音频和加噪音频
- **影响：** 每个batch的preprocessing时间翻倍
- **缓解：** 使用更大的batch_size来分摊开销

### 2. Windows限制
- **num_workers=0：** 数据加载在主进程，较慢
- **解决方案：** 使用WSL2或Linux环境

### 3. 小batch_size
- **影响：** GPU利用率低
- **解决方案：** 增加batch_size（如果内存允许）

## 性能测试方法

### 1. 测量每个batch的时间
```python
import time
start = time.time()
# training step
end = time.time()
print(f"Batch time: {end - start:.3f}s")
```

### 2. 检查GPU利用率
```bash
# Linux/Mac
watch -n 1 nvidia-smi

# Windows
nvidia-smi -l 1
```

**目标：** GPU利用率 > 80%

### 3. 检查数据加载时间
在训练日志中查看：
- Data loading time vs Training time
- 如果数据加载 > 训练时间，增加batch_size或num_workers

## 与NeMo性能对比

### 相同配置下的预期性能

| 配置 | NeMo | 我们的实现 | 差异 |
|------|------|-----------|------|
| 单GPU, batch_size=8 | 基准 | 基准 | 应该相同 |
| 单GPU, batch_size=16 | +50% | +50% | 相同 |
| 4 GPU DDP | +300% | +300% | 相同 |
| bf16-mixed | +100% | +100% | 相同 |

### 如果性能仍然慢

1. **检查数据加载：**
   - 数据集是否在SSD上？
   - `num_workers`是否正确设置？
   - 是否有I/O瓶颈？

2. **检查GPU利用率：**
   - 是否所有GPU都在工作？
   - GPU利用率是否 > 80%？

3. **检查batch_size：**
   - 是否太小？
   - 是否可以增加？

4. **检查DDP配置：**
   - `gradient_as_bucket_view: true`是否设置？
   - `sync_batchnorm: true`是否设置？

5. **检查精度：**
   - 是否使用了混合精度（bf16-mixed）？

## 配置文件示例（高性能）

```yaml
model:
  train_ds:
    batch_size: 16  # 增加batch_size
    num_workers: 8  # Linux/Mac使用，Windows设为0
    pin_memory: true

trainer:
  devices: -1  # 使用所有GPU
  strategy:
    _target_: lightning.pytorch.strategies.DDPStrategy
    gradient_as_bucket_view: true
    static_graph: false
  sync_batchnorm: true
  precision: bf16-mixed  # 如果GPU支持
```

