# DDP 性能优化指南

## 已实施的优化

### 1. ✅ `gradient_as_bucket_view: true`
- **作用：** 使用梯度桶作为视图，减少内存使用
- **性能提升：** 减少内存占用，提高训练速度
- **NeMo对齐：** NeMo所有LLM和VLM模型都使用此配置

### 2. ✅ `sync_batchnorm: true`
- **作用：** 在DDP模式下同步batch normalization
- **重要性：** 确保多GPU训练时batch norm统计量正确

### 3. ✅ `static_graph: false` (默认)
- **作用：** 允许动态计算图
- **说明：** 如果模型结构固定，可以设置为`true`以获得更好性能

## 推荐的DDP配置（Linux/Multi-GPU）

### 基本配置

```yaml
trainer:
  devices: -1  # 使用所有可用GPU
  num_nodes: 1
  accelerator: gpu
  strategy:
    _target_: lightning.pytorch.strategies.DDPStrategy
    gradient_as_bucket_view: true  # 内存优化
    static_graph: false  # 如果模型结构固定，设为true
  sync_batchnorm: true  # DDP训练必需
```

### 高级优化配置

```yaml
trainer:
  devices: -1
  num_nodes: 1
  accelerator: gpu
  strategy:
    _target_: lightning.pytorch.strategies.DDPStrategy
    gradient_as_bucket_view: true
    static_graph: false  # 如果模型结构固定，设为true
    # 可选：增加bucket大小以提高大模型性能
    # bucket_cap_mb: 50  # 默认25MB，可以增加到50-100MB
  sync_batchnorm: true
  precision: bf16-mixed  # 如果GPU支持，使用bf16可以提升2x速度
  accumulate_grad_batches: 1
```

## 性能对比

### 优化前 vs 优化后

| 配置项 | 优化前 | 优化后 | 性能提升 |
|--------|--------|--------|---------|
| `gradient_as_bucket_view` | false (默认) | true | +10-20% 速度，-20-30% 内存 |
| `sync_batchnorm` | false | true | 正确性必需 |
| `precision` | 32 | bf16-mixed | +50-100% 速度 (如果GPU支持) |
| `static_graph` | false | true (如果适用) | +5-10% 速度 |

## 与NeMo对齐

NeMo的所有大型模型训练配置都使用：
- `gradient_as_bucket_view: true`
- `sync_batchnorm: true` (DDP模式下)
- `precision: bf16-mixed` (如果GPU支持)

## 故障排除

### 如果遇到内存不足：
1. 减少 `batch_size`
2. 增加 `accumulate_grad_batches` (梯度累积)
3. 使用 `precision: 16-mixed` 或 `bf16-mixed`

### 如果遇到速度慢：
1. 确保 `gradient_as_bucket_view: true`
2. 如果模型结构固定，设置 `static_graph: true`
3. 使用 `precision: bf16-mixed` (如果GPU支持)
4. 增加 `batch_size` (如果内存允许)
5. 使用SSD存储数据集
6. 增加 `num_workers` (Linux/Mac)

### 如果遇到DDP错误：
1. 确保 `sync_batchnorm: true`
2. 检查所有rank都在处理数据
3. 确保数据加载器正确使用 `DistributedSampler`

## Windows限制

Windows上DDP可能不工作，建议：
- 使用WSL2 (Windows Subsystem for Linux)
- 或使用单GPU训练 (`devices: 1, strategy: auto`)

## 性能监控

### 检查GPU利用率
```bash
# Linux/Mac
watch -n 1 nvidia-smi

# Windows
nvidia-smi -l 1
```

**目标：** 所有GPU利用率应该 > 80%

### 检查数据加载时间
在训练日志中查看：
- Data loading time 应该 < Training time
- 如果数据加载是瓶颈，增加 `batch_size` 或使用SSD

## 预期性能提升

组合所有优化后，预期可以获得：
- **单GPU：** +20-30% 速度提升
- **多GPU (DDP)：** +30-50% 速度提升
- **内存使用：** -20-30% 减少

