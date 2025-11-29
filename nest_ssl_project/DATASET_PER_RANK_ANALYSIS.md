# Dataset Per Rank 分析

## NeMo 原版行为

### 1. 普通 Dataset (非tarred, 非concat)
- **所有rank看到相同的dataset**
- `get_char_dataset` 或 `get_audio_noise_dataset` **不接收** `global_rank` 和 `world_size`
- PyTorch Lightning 的 DistributedSampler 自动处理数据分割
- 所有rank的 `len(dataset)` **应该相同**

### 2. ConcatDataset
- **每个rank看到不同的dataset**
- `get_concat_audio_noise_dataset` **接收** `global_rank` 和 `world_size`
- ConcatDataset 内部根据 `global_rank` 和 `world_size` 分割数据
- 每个rank的 `len(dataset)` **可能不同**（`len // world_size`）

### 3. TarredDataset
- **每个rank看到不同的shards**
- `get_tarred_audio_noise_dataset` **接收** `global_rank` 和 `world_size`
- 根据 `global_rank` 分配不同的tar文件shards
- 每个rank的 `len(dataset)` **可能不同**

## 我们的实现

### 当前状态
- ✅ `get_audio_noise_dataset` 不接收 `global_rank` 和 `world_size`（正确）
- ✅ `get_concat_audio_noise_dataset` 接收 `global_rank` 和 `world_size`（正确）
- ✅ `get_tarred_audio_noise_dataset` 接收 `global_rank` 和 `world_size`（正确）
- ✅ ConcatDataset 的 `__len__` 已对齐：`len(dataset) // world_size`（正确）
- ✅ ConcatDataset 的 `__iter__` 已对齐：根据 `global_rank` 分割数据（正确）

### 验证逻辑
- ✅ 对于普通dataset：验证所有rank看到相同长度
- ✅ 对于ConcatDataset：允许不同rank看到不同长度
- ✅ 对于TarredDataset：允许不同rank看到不同长度

## 关键点

1. **普通dataset**：
   - 所有rank看到**相同的**dataset
   - `batches_per_rank = (dataset_len // world_size) // batch_size`
   - DistributedSampler 处理数据分配

2. **ConcatDataset**：
   - 每个rank看到**不同的**dataset（已分割）
   - `batches_per_rank = dataset_len // batch_size`（dataset_len已经是per-rank）
   - ConcatDataset 内部处理数据分配

3. **TarredDataset**：
   - 每个rank看到**不同的**shards
   - `batches_per_rank` 需要根据实际dataset_len计算
   - TarredDataset 内部处理数据分配

## 当前实现状态

✅ **已对齐NeMo原版**：
- ConcatDataset 的 `__len__` 和 `__iter__` 已对齐
- 验证逻辑已区分普通dataset和ConcatDataset
- batch计算逻辑已区分普通dataset和ConcatDataset

## 调试建议

训练时应该看到：
- **普通dataset**：
  ```
  Verified: All ranks have same dataset_len=1000 (rank 0)
  DDP Training: dataset_len=1000, batches_per_rank=31, is_concat=False
  ```

- **ConcatDataset**：
  ```
  Using ConcatDataset: Rank 0 has dataset_len=250 (different ranks may have different lengths, this is expected)
  DDP Training: dataset_len=250, batches_per_rank=31, is_concat=True
  ```

