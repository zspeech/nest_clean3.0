# 精度对齐运行指南

本目录包含用于运行 NeMo 和 nest_ssl_project 精度对齐的脚本文件。

## 脚本文件说明

### Windows 用户
- `nemo.bat` - 运行 NeMo 训练并保存输出
- `nest_train.bat` - 运行 nest_ssl_project 训练并保存输出
- `compare_outputs.bat` - 比较两个框架的输出

### Linux/Mac 用户
- `nemo.sh` - 运行 NeMo 训练并保存输出
- `nest_train.sh` - 运行 nest_ssl_project 训练并保存输出
- `compare_outputs.sh` - 比较两个框架的输出

## 快速开始

### 步骤 1: 运行 NeMo 训练

**Windows:**
```bash
nemo.bat
```

**Linux/Mac:**
```bash
chmod +x nemo.sh
./nemo.sh
```

这将：
- 使用 dummy_ssl 数据运行 NeMo 训练
- 保存输出到 `nest_ssl_project/saved_nemo_outputs/`
- 只运行第一个 batch（用于快速测试）

### 步骤 2: 运行 nest_ssl_project 训练

**Windows:**
```bash
nest_train.bat
```

**Linux/Mac:**
```bash
chmod +x nest_train.sh
./nest_train.sh
```

这将：
- 使用相同的 dummy_ssl 数据运行 nest_ssl_project 训练
- 保存输出到 `nest_ssl_project/saved_nest_outputs/`
- 使用相同的随机种子（42）确保可复现性

### 步骤 3: 比较输出

**Windows:**
```bash
compare_outputs.bat
```

**Linux/Mac:**
```bash
chmod +x compare_outputs.sh
./compare_outputs.sh
```

这将比较两个框架的输出并显示：
- 模型结构对齐情况
- 前向输出匹配情况
- Loss 匹配情况
- 参数梯度匹配情况
- 层输出匹配情况

## 输出目录

- `nest_ssl_project/saved_nemo_outputs/` - NeMo 训练输出
- `nest_ssl_project/saved_nest_outputs/` - nest_ssl_project 训练输出

每个输出目录包含：
- `model_structure.pkl` - 模型结构信息
- `step_0/` - 步骤 0 的详细输出
  - `batch.pt` - 输入批次
  - `forward_output.pt` - 前向输出
  - `loss.pt` - 损失值
  - `parameter_gradients.pt` - 参数梯度
  - `parameter_weights.pt` - 参数权重
  - `layer_outputs.pkl` - 所有层的输出

## 自定义配置

如果需要修改配置，可以编辑脚本文件中的参数：

### 修改数据路径
```bash
model.train_ds.manifest_filepath=你的路径/train_manifest.json
model.validation_ds.manifest_filepath=你的路径/val_manifest.json
```

### 修改输出目录
```bash
output_dir=你的输出目录
```

### 修改随机种子
```bash
seed=42  # 确保 NeMo 和 nest_ssl_project 使用相同的种子
```

### 修改保存的步骤
```bash
save_steps="0,1,2"  # 保存多个步骤
```

### 修改训练参数
```bash
trainer.devices=1  # GPU 数量
trainer.max_epochs=1  # 最大 epoch 数
+trainer.limit_train_batches=1  # 限制训练的 batch 数（注意：使用 + 前缀添加新字段）
```

**注意**: 
- 如果配置文件中**没有定义**某个字段，需要使用 `+` 前缀来添加新字段。例如 `+trainer.limit_train_batches=1`、`+output_dir=...`、`+seed=42`。
- 如果配置文件中**已经定义**了某个字段，可以直接覆盖，不需要 `+` 前缀。例如 `trainer.devices=1`、`trainer.max_epochs=1`。

**NeMo 脚本**: 由于 NeMo 的配置文件中没有定义 `output_dir`、`seed`、`save_steps` 等字段，所以需要使用 `+` 前缀。
**nest_ssl_project 脚本**: 由于 nest_ssl_project 的配置文件中已经定义了这些字段，所以可以直接覆盖。

## 注意事项

1. **确保使用相同的随机种子**: NeMo 和 nest_ssl_project 必须使用相同的 seed（默认 42）
2. **确保使用相同的数据**: 两个脚本应该使用相同的数据集
3. **确保配置一致**: 模型配置参数应该完全一致
4. **Windows 用户**: 如果遇到路径问题，可以使用绝对路径
5. **GPU 内存**: 如果遇到 OOM，可以减少 batch_size

## 故障排除

### 路径错误
如果遇到路径错误，检查：
- 数据文件是否存在
- 路径是否正确（Windows 使用反斜杠或正斜杠都可以）
- 脚本是否在正确的目录运行

### GPU 内存不足
减少 batch_size：
```bash
model.train_ds.batch_size=4
```

### 数值不匹配
检查：
- 随机种子是否相同
- 配置参数是否一致
- 数据是否相同

## 详细文档

更多详细信息请参考：
- `nest_ssl_project/tools/ALIGNMENT_GUIDE.md` - 完整的对齐指南

