# Simple Run Commands - Copy and Paste

## NeMo Training with Output Saver

### Windows PowerShell (从 nest_ssl_project 目录运行)

```powershell
# 注意：使用 + 前缀添加配置文件中不存在的键
# 使用绝对路径确保 NeMo 能找到文件
python tools/nemo_training_with_saver.py --config-path C:/Users/zhile/Desktop/Nemo_nest/NeMo/examples/asr/conf/ssl/nest --config-name nest_fast-conformer model.train_ds.manifest_filepath=C:/Users/zhile/Desktop/Nemo_nest/nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=C:/Users/zhile/Desktop/Nemo_nest/nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 +trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

**说明**：
- 配置文件已设置默认值：`output_dir=./saved_nemo_outputs`, `seed=42`, `save_steps="0,1,2,3,4"`
- 无需在命令行中指定这些参数
- 如需覆盖，可以添加：`output_dir=./custom_output seed=100`

## nest_ssl_project Training with Comparator

### Windows PowerShell (从 nest_ssl_project 目录运行)

```powershell
# 注意：使用 + 前缀添加配置文件中不存在的键
python tools/nest_training_with_comparator.py --config-path config --config-name nest_fast-conformer model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 +trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

**说明**：
- 配置文件已设置默认值：`saved_outputs_dir=./saved_nemo_outputs`, `comparison_output_dir=./comparison_results`, `seed=42`, `atol=1e-5`, `rtol=1e-5`
- 无需在命令行中指定这些参数
- 确保先运行 NeMo 训练保存输出，然后再运行此命令进行比较

## 完整工作流程

### 步骤 1: NeMo 环境 - 保存输出

```powershell
# 在 NeMo 环境中，从 nest_ssl_project 目录运行
# 注意：使用 + 前缀添加配置文件中不存在的键
# 使用绝对路径确保 NeMo 能找到文件
python tools/nemo_training_with_saver.py --config-path C:/Users/zhile/Desktop/Nemo_nest/NeMo/examples/asr/conf/ssl/nest --config-name nest_fast-conformer model.train_ds.manifest_filepath=C:/Users/zhile/Desktop/Nemo_nest/nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=C:/Users/zhile/Desktop/Nemo_nest/nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 +trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

### 步骤 2: nest_ssl_project 环境 - 比较输出

```powershell
# 在 nest_ssl_project 环境中运行
# 注意：使用 + 前缀添加配置文件中不存在的键
python tools/nest_training_with_comparator.py --config-path config --config-name nest_fast-conformer model.train_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/train_manifest.json model.train_ds.noise_manifest=null model.validation_ds.manifest_filepath=nest_ssl_project/data/dummy_ssl/val_manifest.json trainer.devices=1 trainer.max_epochs=1 +trainer.limit_train_batches=5 trainer.num_sanity_val_steps=0
```

## 配置文件位置

所有默认值在 `nest_ssl_project/config/nest_fast-conformer.yaml` 文件中：

```yaml
output_dir: ./saved_nemo_outputs
seed: 42
save_steps: "0,1,2,3,4"
saved_outputs_dir: ./saved_nemo_outputs
comparison_output_dir: ./comparison_results
atol: 1e-5
rtol: 1e-5
```

如需修改，直接编辑配置文件即可。

