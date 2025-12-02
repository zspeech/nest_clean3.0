# NeMo 与 nest_ssl_project 精度对齐指南

本指南说明如何使用提供的工具进行 NeMo 和 nest_ssl_project 之间的精度对齐。

## 概述

精度对齐工具包括：
1. **NeMo 训练脚本** (`masked_token_pred_pretrain_with_saver.py`) - 在 NeMo 框架下运行训练并保存输出
2. **nest_ssl_project 训练脚本** (`train_with_saver.py`) - 在 nest_ssl_project 框架下运行训练并保存输出
3. **比较脚本** (`compare_nemo_nest_outputs.py`) - 比较两个框架的输出

## 保存的内容

每个训练脚本会保存以下内容：
- **模型结构** (`model_structure.pkl`) - 所有层的名称、参数形状、数据类型等
- **前向输出** (`forward_output.pt`) - 模型的前向传播输出（log_probs）
- **Loss** (`loss.pt`) - 损失值
- **参数梯度** (`parameter_gradients.pt`) - 所有参数的梯度（通过 `param.grad` 获取）
- **所有层的输出** (`layer_outputs.pkl`) - 每个层的输入和输出（仅前向传播）
- **参数权重** (`parameter_weights.pt`) - 优化器更新后的参数权重（可选）

**注意**: 为了避免 PyTorch 的 inplace 修改错误（特别是与 ReLU 等操作的冲突），我们禁用了 backward hook，只保存参数梯度（通过 `param.grad`）。这已经足够进行精度对齐检查。

## 使用步骤

### 步骤 1: 运行 NeMo 训练并保存输出

在 NeMo 环境中运行：

```bash
cd NeMo
python examples/asr/speech_pretraining/masked_token_pred_pretrain_with_saver.py \
    --config-path examples/asr/conf/ssl/nest \
    --config-name nest_fast-conformer \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.train_ds.noise_manifest=<path to noise manifest> \
    model.validation_ds.manifest_filepath=<path to val manifest> \
    model.validation_ds.noise_manifest=<path to noise manifest> \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=1 \
    output_dir=./saved_nemo_outputs \
    seed=42 \
    save_steps="0"
```

**路径说明：**
- 如果 manifest 路径以 `nest_ssl_project/` 开头，脚本会自动解析为绝对路径
- 也可以使用绝对路径，例如：`C:/Users/zhile/Desktop/Nemo_nest/nest_ssl_project/data/dummy_ssl/train_manifest.json`
- 或者相对于项目根目录的路径

**重要参数说明：**
- `output_dir`: 输出保存目录
- `seed`: 随机种子（必须与 nest_ssl_project 使用相同的种子）
- `save_steps`: 要保存的步骤列表（例如 "0" 表示只保存第一步）
- `trainer.limit_train_batches=1`: 限制只运行一个batch（用于快速测试）

### 步骤 2: 运行 nest_ssl_project 训练并保存输出

在 nest_ssl_project 环境中运行：

```bash
cd nest_ssl_project
python train_with_saver.py \
    --config-path config \
    --config-name nest_fast-conformer \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.train_ds.noise_manifest=<path to noise manifest> \
    model.validation_ds.manifest_filepath=<path to val manifest> \
    model.validation_ds.noise_manifest=<path to noise manifest> \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=1 \
    output_dir=./saved_nest_outputs \
    seed=42 \
    save_steps="0"
```

**重要参数说明：**
- `output_dir`: 输出保存目录（应该与 NeMo 不同）
- `seed`: 随机种子（**必须与 NeMo 使用相同的种子**）
- `save_steps`: 要保存的步骤列表（应该与 NeMo 相同）

### 步骤 3: 比较输出

运行比较脚本：

```bash
cd nest_ssl_project
python tools/compare_nemo_nest_outputs.py \
    --nemo_output_dir ./saved_nemo_outputs \
    --nest_output_dir ./saved_nest_outputs \
    --step 0 \
    --atol 1e-5 \
    --rtol 1e-5 \
    --save_comparison ./comparison_results.pkl
```

**参数说明：**
- `--nemo_output_dir`: NeMo 输出目录
- `--nest_output_dir`: nest_ssl_project 输出目录
- `--step`: 要比较的步骤（默认: 0）
- `--atol`: 绝对容差（默认: 1e-5）
- `--rtol`: 相对容差（默认: 1e-5）
- `--save_comparison`: 保存比较结果的文件路径（可选）

## 比较结果说明

比较脚本会输出以下信息：

### 1. 模型结构比较
- 参数总数对比
- 共同参数数量
- 形状匹配情况
- 数据类型匹配情况

### 2. 前向输出比较
- 是否匹配
- 最大绝对差异
- 平均差异
- 标准差

### 3. Loss 比较
- 是否匹配
- 最大绝对差异
- NeMo 和 nest_ssl_project 的 loss 值

### 4. 参数梯度比较
- 匹配的参数数量
- 匹配率百分比
- 前10个不匹配的参数的详细信息

### 5. 层输出比较
- 匹配的层数量
- 匹配率百分比
- 前10个不匹配的层的详细信息

## 对齐检查清单

确保以下内容一致：

- [ ] **随机种子**: NeMo 和 nest_ssl_project 使用相同的 seed（例如 42）
- [ ] **数据**: 使用相同的数据集和 manifest 文件
- [ ] **配置**: 模型配置参数完全一致（d_model, n_heads, n_layers 等）
- [ ] **批次大小**: 使用相同的 batch_size
- [ ] **优化器**: 使用相同的优化器配置（lr, betas, weight_decay 等）
- [ ] **设备**: 确保在相同的设备上运行（CPU 或 GPU）

## 常见问题

### Q: 为什么某些层的输出不匹配？

A: 可能的原因：
1. 随机种子不一致
2. 数据加载顺序不同
3. 模型初始化不同
4. 数值精度问题（尝试调整 atol/rtol）

### Q: 如何提高对齐精度？

A: 
1. 确保使用相同的随机种子
2. 使用相同的批次大小和数据
3. 检查配置参数是否完全一致
4. 使用更高的数值精度（float64 而不是 float32）

### Q: 梯度不匹配怎么办？

A: 
1. 确保前向输出匹配（梯度依赖于前向输出）
2. 检查 loss 计算是否一致
3. 确保 backward 传播的数值稳定性
4. 检查是否有 dropout 或其他随机操作

## 输出文件结构

```
saved_nemo_outputs/
├── model_structure.pkl          # 模型结构信息
├── metadata.pkl                  # 元数据
└── step_0/                       # 步骤 0 的输出
    ├── batch.pt                  # 输入批次
    ├── forward_output.pt         # 前向输出
    ├── loss.pt                   # 损失值
    ├── parameter_gradients.pt    # 参数梯度
    ├── parameter_weights.pt      # 参数权重（优化器更新后）
    ├── layer_outputs.pkl         # 所有层的输出
    └── metadata.pkl               # 步骤元数据
```

## 注意事项

1. **内存使用**: 保存所有层的输出可能会占用大量内存，建议只保存必要的步骤
2. **磁盘空间**: 确保有足够的磁盘空间存储输出文件
3. **运行时间**: 添加 hooks 和保存操作会增加训练时间
4. **DDP 模式**: 在 DDP 模式下，确保只从 rank 0 保存输出

## 示例命令（使用 dummy 数据）

```bash
# NeMo (从 NeMo 目录运行)
cd NeMo
python examples/asr/speech_pretraining/masked_token_pred_pretrain_with_saver.py \
    --config-path examples/asr/conf/ssl/nest \
    --config-name nest_fast-conformer \
    model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json \
    model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=1 \
    output_dir=./saved_nemo_outputs \
    seed=42 \
    save_steps="0"

# 或者使用绝对路径（Windows）
python examples/asr/speech_pretraining/masked_token_pred_pretrain_with_saver.py \
    --config-path examples/asr/conf/ssl/nest \
    --config-name nest_fast-conformer \
    model.train_ds.manifest_filepath=C:/Users/zhile/Desktop/Nemo_nest/nest_ssl_project/data/dummy_ssl/train_manifest.json \
    model.validation_ds.manifest_filepath=C:/Users/zhile/Desktop/Nemo_nest/nest_ssl_project/data/dummy_ssl/val_manifest.json \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=1 \
    output_dir=./saved_nemo_outputs \
    seed=42 \
    save_steps="0"

# nest_ssl_project (从 nest_ssl_project 目录运行)
cd nest_ssl_project
python train_with_saver.py \
    --config-path config \
    --config-name nest_fast-conformer \
    model.train_ds.manifest_filepath=data/dummy_ssl/train_manifest.json \
    model.validation_ds.manifest_filepath=data/dummy_ssl/val_manifest.json \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=1 \
    output_dir=./saved_nest_outputs \
    seed=42 \
    save_steps="0"

# 比较
python tools/compare_nemo_nest_outputs.py \
    --nemo_output_dir ./saved_nemo_outputs \
    --nest_output_dir ./saved_nest_outputs \
    --step 0 \
    --atol 1e-5 \
    --rtol 1e-5
```

