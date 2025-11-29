# Batch 141 Hanging Debug

## 问题描述
- 之前卡在 batch 71，修复 `drop_last=True` 后，现在卡在 batch 141
- 不同 rank 卡在不同 batch，说明数据分布仍不均

## 可能原因
1. **数据集长度不均**：即使 `drop_last=True`，如果数据集长度不是 `batch_size * world_size` 的整数倍，不同 rank 仍可能处理不同数量的 batch
2. **DistributedSampler 行为**：PyTorch Lightning 的 DistributedSampler 可能在某些情况下不能完全保证所有 rank 的 batch 数量一致
3. **IterableDataset 问题**：如果使用 IterableDataset（如 tarred dataset），`drop_last` 的行为可能不同

## NeMo 原版处理方式
- NeMo 原版不强制 `drop_last=True`，而是依赖 PyTorch Lightning 的 DistributedSampler
- 在 `setup_training_data` 中调整 `limit_train_batches`，确保所有 rank 的 batch 数量一致：
  ```python
  if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
      self._trainer.limit_train_batches = int(
          self._trainer.limit_train_batches
          * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
      )
  ```

## 解决方案
1. **对齐 NeMo 原版**：不强制 `drop_last=True`，而是依赖 PyTorch Lightning 的 DistributedSampler
2. **确保 `limit_train_batches` 正确设置**：在 `setup_training_data` 中调整 `limit_train_batches`，确保所有 rank 的 batch 数量一致
3. **检查数据集长度**：确保数据集长度是 `batch_size * world_size` 的整数倍

## 下一步
- 检查 NeMo 原版是否在 DataLoader 中显式设置 sampler
- 检查 PyTorch Lightning 的 DistributedSampler 是否完全保证所有 rank 的 batch 数量一致
- 如果问题持续，考虑显式设置 DistributedSampler 并确保 `drop_last=True`

