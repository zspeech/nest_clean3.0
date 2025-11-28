# DataLoader性能优化指南

## 🚀 优化概述

针对训练速度慢和显存利用率低的问题，已对DataLoader进行以下优化：

### 主要优化项

1. **增加num_workers**: 从0增加到8（Linux/Mac），显著提升IO并行度
2. **启用persistent_workers**: 保持worker进程存活，减少epoch间重启开销
3. **增加prefetch_factor**: 从默认2增加到4，提高GPU利用率
4. **增加batch_size**: 从2增加到8，提高显存利用率和训练吞吐量

---

## 📊 优化前后对比

| 配置项 | 优化前 | 优化后 | 影响 |
|--------|--------|--------|------|
| `num_workers` | 0 | 8 | ⬆️ IO并行度，显著提升数据加载速度 |
| `persistent_workers` | False | True | ⬆️ 减少epoch间重启开销 |
| `prefetch_factor` | 2 (默认) | 4 | ⬆️ GPU利用率，减少GPU等待时间 |
| `batch_size` | 2 | 8 | ⬆️ 显存利用率，提高训练吞吐量 |
| `pin_memory` | True | True | ✅ 保持不变 |

---

## 🔧 配置说明

### num_workers

```yaml
num_workers: 8  # Linux/Mac推荐值
num_workers: 0  # Windows必须设置为0（multiprocessing限制）
```

**作用**: 
- 并行加载数据的worker进程数
- `num_workers=0`: 单线程加载（主进程），IO瓶颈严重
- `num_workers=8`: 8个并行worker，显著提升IO吞吐量

**推荐值**:
- Linux/Mac: 8-16（根据CPU核心数）
- Windows: 0（multiprocessing限制）

### persistent_workers

```yaml
persistent_workers: true  # 当num_workers > 0时启用
```

**作用**:
- 保持worker进程在epoch之间存活
- 避免每个epoch重新创建worker的开销
- 使用更多内存，但显著提升性能

**要求**: `num_workers > 0`

### prefetch_factor

```yaml
prefetch_factor: 4  # 每个worker预取的batch数
```

**作用**:
- 每个worker预取的batch数量
- 更高的值 = 更好的GPU利用率，但使用更多CPU内存
- 默认值: 2
- 推荐值: 4-8（根据CPU内存）

**内存占用**: `num_workers * prefetch_factor * batch_size * sample_size`

### batch_size

```yaml
batch_size: 8  # 根据GPU显存调整
```

**作用**:
- 每个batch的样本数
- 更大的batch_size = 更高的显存利用率，更快的训练速度
- 需要根据GPU显存调整

**推荐值**:
- 小GPU (8GB): 4-8
- 中GPU (16GB): 8-16
- 大GPU (24GB+): 16-32

---

## ⚡ 性能提升预期

### IO性能提升

- **num_workers=0 → 8**: **3-5x** IO吞吐量提升
- **persistent_workers=True**: **10-20%** epoch切换时间减少
- **prefetch_factor=2 → 4**: **20-30%** GPU利用率提升

### 显存利用率提升

- **batch_size=2 → 8**: **4x** 显存利用率提升
- 更充分利用GPU显存，减少GPU空闲时间

### 总体训练速度

预期总体训练速度提升: **2-4x**

---

## 🛠️ 使用建议

### Linux/Mac环境（推荐配置）

```yaml
train_ds:
  batch_size: 8
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
```

### Windows环境（受限配置）

```yaml
train_ds:
  batch_size: 8  # 可以通过增加batch_size补偿
  num_workers: 0  # Windows限制
  pin_memory: true
  persistent_workers: false  # 需要num_workers > 0
  prefetch_factor: null  # 需要num_workers > 0
```

### 高显存GPU（24GB+）

```yaml
train_ds:
  batch_size: 16-32  # 根据显存调整
  num_workers: 16  # 更多worker
  prefetch_factor: 8  # 更高预取
```

### 低显存GPU（8GB）

```yaml
train_ds:
  batch_size: 4  # 减少batch_size避免OOM
  num_workers: 8
  prefetch_factor: 2  # 减少预取节省内存
```

---

## 🔍 监控和调试

### 检查GPU利用率

```bash
nvidia-smi -l 1  # 每秒刷新一次
```

**期望**: GPU利用率应该接近100%，不应该有长时间的空闲

### 检查IO性能

```bash
# 查看数据加载是否成为瓶颈
# 如果GPU利用率低且IO等待时间长，需要增加num_workers
```

### 检查显存使用

```bash
nvidia-smi  # 查看显存使用情况
```

**期望**: 显存使用应该接近GPU容量（留10-20%余量）

---

## ⚠️ 注意事项

1. **Windows限制**: Windows上`num_workers`必须为0，只能通过增加`batch_size`来补偿
2. **内存占用**: `persistent_workers`和`prefetch_factor`会增加CPU内存占用
3. **显存溢出**: 如果出现OOM错误，减少`batch_size`或`prefetch_factor`
4. **CPU核心数**: `num_workers`不应超过CPU核心数

---

## 📈 进一步优化建议

1. **使用tarred数据集**: 对于大规模数据集，使用tarred格式可以进一步提升IO性能
2. **使用混合精度训练**: `precision: 16-mixed`可以提升训练速度并减少显存占用
3. **使用梯度累积**: 如果显存不足，可以使用`accumulate_grad_batches`模拟更大的batch_size
4. **数据预处理缓存**: 对于重复使用的数据集，考虑预处理并缓存

---

**更新日期**: 2025-01-XX  
**版本**: 1.0

