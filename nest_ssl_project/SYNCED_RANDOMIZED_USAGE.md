# Synced Randomized Strategy 使用说明

## ✅ 已实现

`synced_randomized` bucketing strategy 已完全对齐 NeMo 原版实现。

## 📋 工作原理

### 1. 何时生效

`synced_randomized` strategy **只在以下情况生效**：
- ✅ **Tarred datasets（多个buckets）**：当使用多个tarred datasets时，会调用 `get_chain_dataset`，应用 `synced_randomized` strategy
- ❌ **单个dataset**：如果只有一个dataset，`get_chain_dataset` 会直接返回该dataset，不应用bucketing strategy
- ❌ **Concat datasets**：使用 `ConcatDataset`，不应用 `synced_randomized` strategy

### 2. 实现细节

```python
# 在 get_chain_dataset 中：
bucketing_strategy = ds_config.get('bucketing_strategy', 'synced_randomized')
if bucketing_strategy == 'synced_randomized':
    return audio_to_text.RandomizedChainDataset(datasets=datasets, rnd_seed=0)
```

**关键点**：
- `rnd_seed=0`：所有rank使用相同的随机种子
- 确保所有rank看到相同的数据顺序
- 每个epoch会重新随机化bucket的顺序

### 3. 配置

当前配置（`nest_fast-conformer.yaml`）：
```yaml
train_ds:
  bucketing_strategy: "synced_randomized"
  bucketing_batch_size: null
```

## 🎯 使用场景

### 场景1：单个Manifest文件（当前配置）
- **状态**：`synced_randomized` 不会生效
- **原因**：只有一个dataset，不需要bucketing strategy
- **同步保证**：依赖 `drop_last=True` 和 PyTorch Lightning 的 DistributedSampler

### 场景2：多个Tarred Datasets（Bucketing）
- **状态**：`synced_randomized` **会生效**
- **配置示例**：
  ```yaml
  train_ds:
    is_tarred: true
    manifest_filepath: [[bucket1/manifest.json], [bucket2/manifest.json], [bucket3/manifest.json]]
    tarred_audio_filepaths: [[bucket1/tars/*.tar], [bucket2/tars/*.tar], [bucket3/tars/*.tar]]
    bucketing_strategy: "synced_randomized"
    bucketing_batch_size: null
  ```
- **效果**：
  - 所有rank在每个epoch看到相同的bucket顺序
  - 每个epochbucket顺序会重新随机化
  - 确保DDP同步

### 场景3：Concat Datasets
- **状态**：`synced_randomized` 不会生效
- **原因**：使用 `ConcatDataset`，有自己的sampling机制
- **同步保证**：`ConcatDataset` 内部处理同步

## 🔍 验证 `synced_randomized` 是否生效

### 检查方法

1. **检查日志**：
   ```
   Batch bucketing is enabled for N buckets with fixed batch size of X!
   ```

2. **检查dataset类型**：
   ```python
   print(type(dataset))  # 应该是 RandomizedChainDataset
   ```

3. **检查随机种子**：
   - 所有rank应该使用相同的随机种子（0）
   - 每个epoch会重新随机化

## ⚠️ 注意事项

1. **单个dataset**：
   - `synced_randomized` 不会生效
   - 这是正常的，因为不需要bucketing strategy
   - 同步依赖 `drop_last=True` 和 DistributedSampler

2. **多个buckets**：
   - `synced_randomized` 会确保所有rank看到相同的bucket顺序
   - 有助于DDP同步
   - 但不能完全解决batch数量不一致的问题（如果数据集长度不是 `batch_size * world_size` 的整数倍）

3. **与 `drop_last` 的关系**：
   - `synced_randomized` 确保bucket顺序一致
   - `drop_last=True` 确保batch数量一致
   - 两者结合使用效果最好

## 📊 当前状态

- ✅ `synced_randomized` 实现已对齐 NeMo
- ✅ 配置文件中已设置 `bucketing_strategy: "synced_randomized"`
- ⚠️ 当前使用单个manifest文件，`synced_randomized` 不会生效
- ✅ 如果将来使用多个buckets，`synced_randomized` 会自动生效

## 🎯 总结

`synced_randomized` strategy **已正确实现**，会在使用多个tarred datasets（bucketing）时自动生效。对于当前单个manifest文件的配置，这是正常的，因为不需要bucketing strategy。同步主要依赖 `drop_last=True` 和 PyTorch Lightning 的 DistributedSampler。

