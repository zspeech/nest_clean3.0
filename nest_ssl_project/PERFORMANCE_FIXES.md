# 性能优化修复说明

## 已修复的问题

### 1. ✅ 配置文件缺少性能参数
**问题：** 配置文件中没有 `persistent_workers` 和 `prefetch_factor` 参数
**修复：** 已在配置文件中添加这些参数，并添加了注释说明如何在Linux/Mac上启用

### 2. ✅ 代码已支持但配置未启用
**问题：** 代码中已经添加了 `persistent_workers` 和 `prefetch_factor` 支持，但配置文件中没有设置
**修复：** 现在配置文件中已添加，当 `num_workers > 0` 时会自动启用

### 3. ⚠️ 主要性能瓶颈：num_workers: 0
**问题：** Windows兼容性考虑，`num_workers` 设置为0，导致单进程数据加载
**影响：** 这是最大的性能瓶颈，GPU会等待CPU处理数据
**解决方案：**
- **Windows平台：** 保持 `num_workers: 0`（Windows多进程有问题）
- **Linux/Mac平台：** 取消注释配置中的 `num_workers: 4` 行，并设置 `persistent_workers: true`

### 4. ⚠️ Batch Size太小
**问题：** `batch_size: 2` 太小，GPU利用率低
**建议：** 根据GPU内存增加batch size（例如8, 16, 32）

### 5. ⚠️ Forward方法调用两次Preprocessor
**问题：** `EncDecDenoiseMaskedTokenPredModel.forward()` 中需要处理两个信号：
- `input_signal` → `processed_signal`（用于生成tokens）
- `noisy_input_signal` → `processed_noisy_input_signal`（用于编码）

**说明：** 这是模型架构的要求，不是bug。但这是性能瓶颈之一。

## 性能优化建议

### 立即可用的优化（Windows）

1. **增加batch size**
   ```yaml
   batch_size: 8  # 或16，根据GPU内存调整
   ```

2. **启用pin_memory**（已启用）
   ```yaml
   pin_memory: true
   ```

3. **使用tarred数据集**（如果可能）
   ```yaml
   is_tarred: true
   tarred_audio_filepaths: ["path/to/tarred/files"]
   ```

### Linux/Mac平台优化

1. **启用多进程数据加载**
   ```yaml
   num_workers: 4  # 根据CPU核心数调整（4-8）
   persistent_workers: true
   prefetch_factor: 2
   ```

2. **增加batch size**
   ```yaml
   batch_size: 16  # 或更大
   ```

3. **启用benchmark模式**（如果输入尺寸固定）
   ```yaml
   trainer:
     benchmark: true
   ```

## 性能对比预期

| 配置 | 预期吞吐量提升 |
|------|---------------|
| Windows (num_workers=0) | 基准 |
| Windows + batch_size=8 | +50-100% |
| Linux/Mac (num_workers=4) | +200-400% |
| Linux/Mac + batch_size=16 | +300-500% |

## 下一步

1. **测试优化后的性能**
2. **如果仍在Windows上，考虑：**
   - 使用WSL2运行训练
   - 增加batch size
   - 使用更快的存储（SSD）
3. **如果可能，迁移到Linux/Mac平台进行训练**

## 配置文件更改

已更新 `nest_fast-conformer.yaml`：
- ✅ 添加了 `persistent_workers` 配置
- ✅ 添加了 `prefetch_factor` 配置
- ✅ 添加了Linux/Mac平台的注释说明

