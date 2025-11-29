# Batch 141 卡住问题修复

## 问题描述
- 4个GPU在batch 141卡住
- 不同rank卡在不同batch，说明数据分布不均导致DDP同步死锁

## 根本原因
即使`drop_last=True`在配置文件中，如果数据集长度不是`batch_size * world_size`的整数倍，PyTorch Lightning的DistributedSampler仍可能导致不同rank处理不同数量的batch。

## 修复方案

### 1. 强制`drop_last=True`（代码级别）
```python
# 在 _setup_dataloader_from_config 中
drop_last = config.get('drop_last', False)
if self.world_size > 1:
    # 强制drop_last=True，确保所有rank处理相同数量的batch
    drop_last = True
```

### 2. 添加详细的batch数量日志
```python
# 在 setup_training_data 中
batches_per_rank = (dataset_len // self.world_size) // batch_size
total_batches = batches_per_rank * self.world_size
logging.info(
    f"DDP Training: dataset_len={dataset_len}, batch_size={batch_size}, "
    f"world_size={self.world_size}, batches_per_rank={batches_per_rank}, "
    f"total_batches={total_batches}"
)
```

### 3. 改进`limit_train_batches`处理
- 确保`limit_train_batches`不超过`batches_per_rank`
- 如果超过，记录警告
- 正确计算每个rank的batch数量

## 关键更改

1. **强制drop_last=True**：当`world_size > 1`时，代码强制设置`drop_last=True`，即使配置文件中是`False`
2. **详细日志**：添加了详细的batch数量日志，方便调试
3. **limit_train_batches验证**：确保`limit_train_batches`不超过每个rank的batch数量

## 验证方法

训练开始时应该看到类似日志：
```
DDP Training: dataset_len=1000, batch_size=8, world_size=4, batches_per_rank=31, total_batches=124
Adjusted limit_train_batches to 31 (batches_per_rank=31)
```

如果所有rank的`batches_per_rank`相同，说明同步应该正常。

## 注意事项

1. **数据集长度**：如果数据集长度不是`batch_size * world_size`的整数倍，最后一个不完整的batch会被丢弃
2. **limit_train_batches**：如果设置过大，可能导致某些rank处理更多batch，导致同步问题
3. **调试**：如果问题仍然存在，检查日志中的`batches_per_rank`是否在所有rank上相同

## 下一步

如果问题仍然存在：
1. 检查日志中所有rank的`batches_per_rank`是否相同
2. 如果不同，检查数据集长度和batch_size
3. 考虑调整数据集大小或batch_size，使`dataset_len % (batch_size * world_size) == 0`

