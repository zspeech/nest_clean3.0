# DDP并行调试指南

## 🔍 问题诊断

### 症状
- 打印/日志只显示一个rank的输出
- 其他rank似乎没有训练

### 可能原因
1. **日志配置**: 只有rank 0在打印（这是正常的，但需要确认所有rank都在训练）
2. **数据分布**: 某些rank没有分配到数据
3. **DDP初始化**: DDP没有正确初始化
4. **训练循环**: 某些rank没有进入训练循环

---

## ✅ 检查清单

### 1. 确认所有rank都在训练

在`training_step`中添加调试输出：

```python
def training_step(self, batch: ssl_dataset.AudioNoiseBatch, batch_idx: int):
    # 调试: 确认所有rank都在训练
    if batch_idx % 100 == 0:  # 每100个batch打印一次
        print(f"[Rank {self.global_rank}] Training step {batch_idx}, batch size: {batch.audio.size(0)}")
    
    # ... 正常训练代码
```

**期望**: 应该看到所有rank的输出（rank 0, 1, 2, ...）

### 2. 检查数据分布

在`setup_training_data`中添加调试输出：

```python
def setup_training_data(self, train_data_config):
    # ... 设置数据加载器
    
    if self._trainer is not None:
        print(f"[Rank {self.global_rank}] World size: {self.world_size}, "
              f"Dataset size: {len(self._train_dl.dataset) if hasattr(self._train_dl, 'dataset') else 'N/A'}, "
              f"Batches per rank: {len(self._train_dl) if self._train_dl else 'N/A'}")
```

**期望**: 每个rank应该看到不同的数据集大小（如果使用DistributedSampler）

### 3. 检查DDP初始化

在`train.py`中添加：

```python
@hydra_runner(config_path="config", config_name="nest_fast-conformer")
def main(cfg):
    import torch.distributed as dist
    
    trainer = pl.Trainer(**cfg.trainer)
    
    # 检查DDP是否初始化
    if dist.is_available() and dist.is_initialized():
        print(f"[Rank {dist.get_rank()}] DDP initialized. World size: {dist.get_world_size()}")
    else:
        print("[Rank 0] DDP not initialized (single GPU or CPU training)")
    
    # ... 继续训练
```

### 4. 检查GPU利用率

```bash
# 在训练时运行
nvidia-smi -l 1
```

**期望**: 所有GPU都应该显示使用率（如果使用多GPU）

---

## 🔧 常见问题和解决方案

### 问题1: 只有rank 0在打印

**原因**: 日志配置只允许rank 0打印（这是正常的）

**解决方案**: 
- 这是**正常行为**，PyTorch Lightning默认只从rank 0打印
- 如果需要所有rank的输出，在代码中明确打印：

```python
def training_step(self, batch, batch_idx):
    # 强制所有rank都打印
    print(f"[Rank {self.global_rank}] Step {batch_idx}")
    # 或者使用logging（会自动处理rank）
    logging.info(f"[Rank {self.global_rank}] Step {batch_idx}")
```

### 问题2: 某些rank没有训练

**检查**:
1. 确认所有rank都进入了`training_step`
2. 检查数据分布是否正确
3. 检查`world_size`是否正确设置

**解决方案**:
```python
def training_step(self, batch, batch_idx):
    # 添加调试输出
    if batch_idx == 0:
        print(f"[Rank {self.global_rank}] First training step, batch shape: {batch.audio.shape}")
    
    # 检查batch是否为空
    if batch.audio.size(0) == 0:
        print(f"[Rank {self.global_rank}] WARNING: Empty batch!")
        return None
    
    # ... 正常训练
```

### 问题3: 数据分布不均匀

**检查**:
```python
def setup_training_data(self, train_data_config):
    # ... 设置数据加载器
    
    # 检查每个rank的数据量
    if hasattr(self._train_dl, 'dataset'):
        dataset_size = len(self._train_dl.dataset)
        batches_per_rank = len(self._train_dl)
        print(f"[Rank {self.global_rank}] Dataset size: {dataset_size}, "
              f"Batches per rank: {batches_per_rank}, "
              f"World size: {self.world_size}")
```

**解决方案**:
- 确保使用`DistributedSampler`（PyTorch Lightning自动处理）
- 检查`drop_last`设置（如果数据不能均匀分配）

### 问题4: DDP没有正确初始化

**检查**:
```python
import torch.distributed as dist

def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    
    # 检查DDP状态
    print(f"DDP available: {dist.is_available()}")
    print(f"DDP initialized: {dist.is_initialized()}")
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
```

**解决方案**:
- 确保使用`strategy="ddp"`或`strategy="auto"`（多GPU时）
- 确保使用正确的启动命令（`torchrun`或`python -m torch.distributed.launch`）

---

## 🚀 调试代码模板

### 在training_step中添加调试

```python
def training_step(self, batch: ssl_dataset.AudioNoiseBatch, batch_idx: int):
    # 调试输出（每N个batch）
    if batch_idx % 100 == 0:
        print(f"[Rank {self.global_rank}/{self.world_size}] "
              f"Step {batch_idx}, Batch size: {batch.audio.size(0)}, "
              f"Loss device: {batch.audio.device}")
    
    # 正常训练代码
    log_probs, encoded_len, masks, tokens = self.forward(...)
    loss_value = self.loss(...)
    
    # 检查loss是否有效
    if torch.isnan(loss_value) or torch.isinf(loss_value):
        print(f"[Rank {self.global_rank}] WARNING: Invalid loss: {loss_value}")
    
    return loss_value
```

### 在setup_training_data中添加调试

```python
def setup_training_data(self, train_data_config):
    # 更新world_size
    if self._trainer is not None:
        self.world_size = self._trainer.world_size
    
    # 调试输出
    print(f"[Rank {self.global_rank}] Setting up training data. "
          f"World size: {self.world_size}, "
          f"Global rank: {self.global_rank}")
    
    # ... 设置数据加载器
    
    if self._train_dl is not None:
        print(f"[Rank {self.global_rank}] Training dataloader created. "
              f"Batches: {len(self._train_dl)}")
```

---

## 📊 验证DDP正常工作

### 1. 检查所有rank都在训练

运行训练，应该看到：
```
[Rank 0/4] Step 0, Batch size: 8
[Rank 1/4] Step 0, Batch size: 8
[Rank 2/4] Step 0, Batch size: 8
[Rank 3/4] Step 0, Batch size: 8
```

### 2. 检查数据分布

每个rank应该处理不同的数据：
- Rank 0: 处理样本 0, 4, 8, ...
- Rank 1: 处理样本 1, 5, 9, ...
- Rank 2: 处理样本 2, 6, 10, ...
- Rank 3: 处理样本 3, 7, 11, ...

### 3. 检查GPU利用率

```bash
nvidia-smi -l 1
```

所有GPU都应该显示使用率。

### 4. 检查训练速度

多GPU训练应该比单GPU快（接近线性加速）：
- 2 GPU: ~1.8-1.9x
- 4 GPU: ~3.5-3.8x
- 8 GPU: ~7.0-7.5x

---

## ⚠️ 注意事项

1. **日志输出**: 默认只有rank 0打印是**正常的**，这避免重复输出
2. **数据分布**: 使用`DistributedSampler`自动处理数据分布
3. **同步**: DDP自动同步梯度，不需要手动同步
4. **验证**: 使用`nvidia-smi`检查所有GPU都在使用

---

## 🔍 快速诊断命令

```bash
# 1. 检查GPU使用情况
nvidia-smi -l 1

# 2. 检查进程
ps aux | grep python

# 3. 检查DDP进程数（应该等于GPU数）
ps aux | grep python | wc -l
```

---

**更新日期**: 2025-01-XX  
**版本**: 1.0

