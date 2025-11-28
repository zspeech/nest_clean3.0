# 安装和使用指南

## ✅ 重要提示

**本项目已经完全独立于 NeMo 框架，不需要安装 NeMo！**

## 📦 快速安装

### 1. 安装基础依赖

```bash
cd nest_ssl_project
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "import torch; import lightning; print('安装成功！')"
```

## 🪟 Windows 特殊说明

### Windows 配置优化

本项目已针对 Windows 环境优化，配置文件已设置：
- `trainer.devices: 1` - Windows 兼容
- `trainer.strategy: auto` - 避免 DDP 问题
- `num_workers: 0` - 避免多进程问题

### Windows 运行

```bash
# 直接运行（已优化配置）
python train.py
```

如果遇到问题，可以手动指定参数：

```bash
python train.py \
    trainer.devices=1 \
    trainer.strategy=auto \
    model.train_ds.num_workers=0 \
    model.validation_ds.num_workers=0
```

## 🚀 快速开始

### 1. 准备数据

项目已包含 dummy 测试数据：
- `data/dummy_ssl/train_manifest.json`
- `data/dummy_ssl/val_manifest.json`

### 2. 运行训练

```bash
# 使用默认配置（dummy 数据）
python train.py

# 指定自己的数据
python train.py \
    model.train_ds.manifest_filepath=/path/to/train.json \
    model.validation_ds.manifest_filepath=/path/to/val.json
```

## 📋 依赖说明

### 核心依赖（必须）

- **torch** >= 2.0.0 - PyTorch 深度学习框架
- **lightning** >= 2.0.0 - PyTorch Lightning 训练框架
- **hydra-core** >= 1.3.0 - 配置管理
- **omegaconf** >= 2.3.0 - 配置解析
- **soundfile** >= 0.12.0 - 音频文件读取
- **librosa** >= 0.10.0 - 音频处理

### 可选依赖

- **wandb** - 实验跟踪（如果使用 WandB）
- **tensorboard** - TensorBoard 日志（如果使用 TensorBoard）

## 🔧 安装 NeMo（仅用于对比）

如果需要与 NeMo 对比，可以安装 NeMo：

### Windows 最小安装（跳过编译问题）

```bash
# 安装核心依赖
pip install torch torchaudio
pip install pytorch-lightning hydra-core omegaconf

# 安装 NeMo（跳过编译问题包）
pip install nemo-toolkit[asr] --no-deps
pip install nemo-toolkit[all] --no-deps

# 手动安装依赖
pip install ruamel.yaml tqdm wget packaging
pip install transformers datasets
```

### 手动安装步骤

按照上面的命令手动安装即可。如果需要自动化脚本，可以参考上面的步骤自行创建。

**注意**: `megatron_core`, `ctc_segmentation`, `texterrors` 等包在 Windows 上可能无法编译，但不影响核心功能。

## ❓ 常见问题

### Q: 内存不足怎么办？

- 减少 `batch_size`
- 使用梯度累积
- 启用混合精度训练

### Q: 如何查看训练日志？

训练日志默认保存在 `experiments/` 目录下，或使用 TensorBoard：

```bash
tensorboard --logdir=experiments
```

### Q: 支持哪些音频格式？

支持常见的音频格式：WAV、MP3、FLAC、OPUS 等。

## 📚 相关文档

- [README.md](README.md) - 项目主文档
- [PROJECT_STRUCTURE_CLEAN.md](PROJECT_STRUCTURE_CLEAN.md) - 项目结构说明
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 快速参考
