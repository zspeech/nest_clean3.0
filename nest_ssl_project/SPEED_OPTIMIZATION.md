# 训练速度优化指南

如果训练速度仍然很慢，请按照以下步骤进行优化：

## 立即可以尝试的优化（按优先级排序）

### 1. ⚡ 增加 Batch Size（最重要）

**当前配置：** `batch_size: 2`  
**推荐配置：** 根据GPU内存调整

```yaml
batch_size: 8   # 对于8GB GPU
batch_size: 16  # 对于16GB GPU
batch_size: 32  # 对于24GB+ GPU
```

**预期提升：** +50-200% 吞吐量

### 2. ⚡ 使用混合精度训练（如果GPU支持）

**当前配置：** `precision: 32`  
**推荐配置：**

```yaml
precision: 16-mixed  # 对于NVIDIA GPU (Tensor Cores)
# 或
precision: bf16-mixed  # 对于更新的GPU (A100, H100等)
```

**预期提升：** +50-100% 吞吐量，几乎不损失精度

**注意：** 如果遇到NaN或训练不稳定，改回 `precision: 32`

### 3. ⚡ 减少日志频率

**当前配置：** `log_every_n_steps: 10`  
**推荐配置：**

```yaml
log_every_n_steps: 50  # 或100，减少日志开销
```

**预期提升：** +5-10% 吞吐量

### 4. ⚡ 使用 Tarred 数据集（如果可能）

如果数据集很大，使用tarred格式可以显著减少I/O时间：

```yaml
is_tarred: true
tarred_audio_filepaths: ["path/to/tarred/files"]
```

**预期提升：** +20-50% 吞吐量（取决于I/O瓶颈）

### 5. ⚡ 减少验证频率

**当前配置：** `val_check_interval: 1.0`  
**推荐配置：**

```yaml
val_check_interval: 0.5  # 每半个epoch验证一次
# 或
val_check_interval: 1000  # 每1000步验证一次
```

**预期提升：** +10-20% 吞吐量

## Windows平台特定优化

由于Windows上 `num_workers: 0` 的限制，以下优化特别重要：

### 1. 增加 Batch Size
这是Windows上最重要的优化：

```yaml
batch_size: 8  # 或更大，根据GPU内存
```

### 2. 使用更快的存储
- 将数据集放在SSD上
- 如果可能，使用NVMe SSD

### 3. 考虑使用WSL2
WSL2允许使用 `num_workers > 0`，可以获得显著性能提升：

```bash
# 在WSL2中
num_workers: 4
persistent_workers: true
prefetch_factor: 2
```

### 4. 减少数据预处理开销
- 使用tarred数据集（预处理的音频）
- 减少 `max_duration` 如果不需要长音频

## Linux/Mac平台优化

### 1. 启用多进程数据加载

```yaml
num_workers: 4  # 或8，根据CPU核心数
persistent_workers: true
prefetch_factor: 2
```

**预期提升：** +200-400% 吞吐量

### 2. 增加 Batch Size

```yaml
batch_size: 16  # 或32
```

### 3. 使用混合精度

```yaml
precision: 16-mixed
```

## 高级优化（需要代码修改）

### 1. 使用 torch.compile() (PyTorch 2.0+)

在模型初始化后添加：

```python
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='reduce-overhead')
```

**预期提升：** +10-30% 吞吐量

### 2. 优化数据加载

- 使用更快的音频解码库（如soundfile而不是librosa）
- 预计算特征（如果可能）

### 3. 减少模型大小（如果精度允许）

- 减少 `d_model`（例如从512到256）
- 减少 `n_layers`（例如从17到12）

## 性能监控

使用以下命令监控GPU利用率：

```bash
# Windows
nvidia-smi -l 1

# Linux/Mac
watch -n 1 nvidia-smi
```

**目标：** GPU利用率应该 > 80%

如果GPU利用率低（< 50%），主要瓶颈是数据加载，需要：
- 增加 `batch_size`
- 使用 `num_workers > 0`（Linux/Mac）
- 使用tarred数据集

## 快速配置切换

已创建快速训练配置文件：`nest_fast-conformer_fast.yaml`

使用方式：
```bash
python train.py --config-path=config --config-name=nest_fast-conformer_fast
```

该配置文件包含：
- ✅ `batch_size: 8`（从2增加）
- ✅ `precision: 16-mixed`（混合精度）
- ✅ `log_every_n_steps: 50`（减少日志频率）
- ✅ `drop_last: true`（一致的batch大小）

## 预期性能提升

| 优化项 | 预期提升 | 难度 |
|--------|---------|------|
| batch_size: 2 → 8 | +100-200% | 简单 |
| precision: 32 → 16-mixed | +50-100% | 简单 |
| num_workers: 0 → 4 (Linux) | +200-400% | 简单 |
| tarred数据集 | +20-50% | 中等 |
| torch.compile() | +10-30% | 中等 |

**组合优化预期：** 在Linux/Mac上，组合所有优化可以获得 **5-10倍** 的性能提升。

## 故障排除

### 如果使用混合精度后出现NaN：

1. 改回 `precision: 32`
2. 或尝试 `precision: bf16-mixed`
3. 检查学习率是否过高

### 如果增加batch_size后OOM：

1. 减少 `batch_size`
2. 使用梯度累积：
   ```yaml
   accumulate_grad_batches: 2  # 相当于batch_size * 2
   ```
3. 减少 `max_duration`

### 如果训练仍然很慢：

1. 检查GPU利用率（应该 > 80%）
2. 检查数据加载时间（使用profiler）
3. 考虑使用更小的模型进行测试

