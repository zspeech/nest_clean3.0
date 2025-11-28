# nest_ssl_project

一个从 NeMo 框架中提取的**完全独立的 SSL 训练项目**，专门用于训练**NEST Fast-Conformer 自监督学习模型**（Denoising Masked Token Prediction）。

## ✨ 核心特点

- ✅ **完全独立**: 不依赖 NeMo 框架，可直接运行
- ✅ **与 NeMo 100% 对齐**: 配置、架构、功能、参数完全一致
  - ✅ DDP配置与NeMo原版一致（`strategy: auto`, `sync_batchnorm: true`）
  - ✅ DataLoader配置与NeMo一致（不使用`persistent_workers`和`prefetch_factor`）
  - ✅ 数据加载逻辑与NeMo一致（`max_trial: 20/100`，默认`librosa.resample`）
  - ✅ 模型架构与NeMo一致（preprocessor双重调用等设计限制）
- ✅ **结构清晰**: 模块化设计，易于理解和维护
- ✅ **Windows 优化**: 已针对 Windows 环境优化配置
- ✅ **功能完整**: 支持完整的 SSL 训练流程

**📊 与 NeMo 对比**: 参见 [COMPARISON.md](COMPARISON.md)  
**📁 项目结构**: 参见 [PROJECT_STRUCTURE_CLEAN.md](PROJECT_STRUCTURE_CLEAN.md)  
**🚀 快速参考**: 参见 [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

## 📋 目录

- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [项目状态](#项目状态)
- [常见问题](#常见问题)
- [许可证](#许可证)

## ✨ 功能特性

- **自监督学习模型**: 实现了 `EncDecDenoiseMaskedTokenPredModel`，用于语音自监督预训练
- **去噪训练**: 支持带噪声的音频数据进行去噪训练
- **掩码 Token 预测**: 实现了掩码语言模型（MLM）风格的训练目标
- **独立运行**: 不依赖完整的 NeMo 框架，可以独立运行
- **简化代码**: 只保留运行训练所需的核心功能

## 📁 项目结构

```
nest_ssl_project/
├── 📄 train.py                    # 主训练脚本
├── 📄 requirements.txt            # 依赖列表
│
├── 📁 config/                      # 配置文件
│   └── nest_fast-conformer.yaml   # NEST Fast-Conformer 配置（与 NeMo 一致）
│
├── 📁 models/                      # 模型定义
│   └── ssl_models.py              # SSL 模型（EncDecDenoiseMaskedTokenPredModel）
│
├── 📁 modules/                     # 神经网络模块
│   ├── conformer_encoder.py       # ConformerEncoder（核心编码器）
│   ├── audio_preprocessing.py     # 音频预处理
│   ├── ssl_modules/               # SSL 专用模块
│   │   ├── quantizers.py         # 向量量化器
│   │   ├── masking.py            # 掩码模块
│   │   ├── multi_softmax_decoder.py  # 多 softmax 解码器
│   │   └── augmentation.py       # 数据增强
│   └── utils/                    # 工具模块
│
├── 📁 data/                        # 数据集
│   ├── ssl_dataset.py             # SSL 数据集
│   ├── audio_to_text.py           # 音频数据集
│   └── dummy_ssl/                 # Dummy 测试数据
│
├── 📁 losses/                      # 损失函数
│   └── ssl_losses/
│       └── mlm.py                 # MLM 损失
│
├── 📁 core/                        # 核心框架（NeMo 替代）
│   ├── classes/                   # 核心类（ModelPT, NeuralModule 等）
│   └── neural_types/              # 类型系统
│
├── 📁 parts/                       # 部分模块
│   ├── preprocessing/             # 预处理工具
│   └── mixins/                    # Mixins
│
├── 📁 utils/                       # 工具函数
│   ├── logging.py                 # 日志
│   ├── exp_manager.py             # 实验管理
│   └── hydra_runner.py            # Hydra 运行器
│
└── 📁 tools/                       # 工具脚本
    ├── prepare_dummy_ssl_data.py   # 生成测试数据
    └── compare_with_nemo.py        # 与 NeMo 对比
```

**详细结构说明**: 参见 [PROJECT_STRUCTURE_CLEAN.md](PROJECT_STRUCTURE_CLEAN.md)  
**与 NeMo 对比**: 参见 [STRUCTURE_COMPARISON.md](STRUCTURE_COMPARISON.md)

## 🚀 安装

### 系统要求

- Python >= 3.8
- CUDA >= 11.0 (如果使用 GPU)
- 足够的磁盘空间用于数据集和模型检查点

### 安装步骤

1. **克隆或下载项目**

```bash
cd nest_ssl_project
```

2. **创建虚拟环境（推荐）**

```bash
# 使用 conda
conda create -n nest_ssl python=3.10
conda activate nest_ssl

# 或使用 venv
python -m venv nest_ssl_env
source nest_ssl_env/bin/activate  # Linux/Mac
nest_ssl_env\Scripts\activate     # Windows
```

3. **安装 PyTorch**

根据你的 CUDA 版本安装 PyTorch：

```bash
# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchaudio
```

4. **安装项目依赖**

```bash
pip install -r requirements.txt
```

详细的安装说明请参考 [INSTALL.md](INSTALL.md)。

## 🏃 快速开始

### 1. 准备数据

准备训练数据的 manifest 文件（JSON 格式），每行一个样本：

```json
{"audio_filepath": "/path/to/audio1.wav", "duration": 10.5, "text": "transcription"}
{"audio_filepath": "/path/to/audio2.wav", "duration": 8.3, "text": "transcription"}
```

同样准备噪声数据的 manifest 文件（可选，用于数据增强）。

### 2. 运行训练

```bash
python train.py \
    model.train_ds.manifest_filepath=/path/to/train_manifest.json \
    model.train_ds.noise_manifest=/path/to/noise_manifest.json \
    model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
    model.validation_ds.noise_manifest=/path/to/noise_manifest.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=100
```

## 📝 配置说明

训练配置通过 Hydra 管理，主要配置文件位于 `config/nest_fast-conformer.yaml`。

### 主要配置项

- **模型配置** (`model`): 模型架构、预处理器、编码器、解码器等
- **数据配置** (`model.train_ds`, `model.validation_ds`): 数据集路径、批次大小等
- **训练配置** (`trainer`): 设备、epochs、学习率等
- **优化器配置** (`model.optim`): 优化器类型、学习率调度等
- **实验管理** (`exp_manager`): 日志、检查点保存等

### 常用配置示例

```bash
# 单 GPU 训练
python train.py \
    model.train_ds.manifest_filepath=train.json \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=50

# 多 GPU 训练（DDP）- 与NeMo配置一致
# Linux/Mac 多 GPU 训练（推荐）
# 注意：默认配置使用 strategy: auto，PyTorch Lightning会自动选择DDP
python train.py \
    model.train_ds.manifest_filepath=train.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="auto" \
    trainer.sync_batchnorm=true \
    trainer.max_epochs=100

# 或显式指定DDP策略（与NeMo其他SSL配置一致）
python train.py \
    model.train_ds.manifest_filepath=train.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.sync_batchnorm=true \
    trainer.max_epochs=100

# Windows 多 GPU 训练（使用 ddp_spawn）
python train.py \
    model.train_ds.manifest_filepath=train.json \
    trainer.devices=2 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp_spawn" \
    trainer.sync_batchnorm=true \
    trainer.max_epochs=100

# 高级 DDP 配置（可选优化，参考 nest_fast-conformer_ddp_example.yaml）
# 注意：PyTorch Lightning 2.0+ 中 find_unused_parameters 参数已被移除
# 默认配置与NeMo原版一致（strategy: auto），如需高级配置请参考示例文件

# 自定义学习率
python train.py \
    model.train_ds.manifest_filepath=train.json \
    model.optim.lr=0.0001 \
    model.optim.sched.warmup_steps=1000
```

## 💡 使用示例

### 基本训练

```bash
python train.py \
    --config-path=config \
    --config-name=nest_fast-conformer \
    model.train_ds.manifest_filepath=data/train_manifest.json \
    model.train_ds.noise_manifest=data/noise_manifest.json \
    model.validation_ds.manifest_filepath=data/val_manifest.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=100
```

### 从检查点恢复训练

```bash
python train.py \
    model.train_ds.manifest_filepath=data/train_manifest.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.max_epochs=200 \
    model.restore_from=/path/to/checkpoint.nemo
```

### 使用 WandB 记录实验

```bash
python train.py \
    model.train_ds.manifest_filepath=data/train_manifest.json \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="my_experiment" \
    exp_manager.wandb_logger_kwargs.project="ssl_pretraining"
```

## 📊 项目状态

**✅ 项目已完成并与 NeMo 100% 对齐！**

当前状态：

- ✅ 核心模型实现完成（与NeMo一致）
- ✅ 数据集加载功能完成（与NeMo一致）
- ✅ 训练脚本可用
- ✅ 所有 NeMo 依赖已移除
- ✅ 项目完全独立运行
- ✅ **配置参数与NeMo原版完全一致**
  - ✅ DDP配置：`strategy: auto`, `sync_batchnorm: true`（与NeMo nest_fast-conformer.yaml一致）
  - ✅ DataLoader配置：基本配置，不使用`persistent_workers`和`prefetch_factor`（与NeMo一致）
  - ✅ 数据加载参数：`max_trial: 20/100`，默认`librosa.resample`（与NeMo一致）
  - ✅ 模型架构：preprocessor双重调用等设计限制（与NeMo一致）
- ✅ 文档完整

**与NeMo对齐确认：**
- ✅ 所有配置参数与NeMo原版`nest_fast-conformer.yaml`一致
- ✅ DDP策略配置与NeMo一致
- ✅ DataLoader配置与NeMo一致
- ✅ 数据加载逻辑与NeMo一致

项目已完全从 NeMo 框架中剥离，可以独立运行，且所有配置与NeMo原版保持一致。详细进度请参考 [PROGRESS.md](PROGRESS.md) 和 [COMPLETION_STATUS.md](COMPLETION_STATUS.md)。

## ❓ 常见问题

### Q: 如何检查 CUDA 是否可用？

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Q: 如何配置使用 CUDA/GPU？

配置文件默认使用 `accelerator: auto`，会自动检测并使用可用的设备（GPU 或 CPU）。

**如果 CUDA 可用**，训练会自动使用 GPU。如果遇到 "No supported gpu backend found!" 错误，可能是：
1. PyTorch 未安装 CUDA 版本：需要重新安装支持 CUDA 的 PyTorch
   ```bash
   # CUDA 11.8
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # CUDA 12.1
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
2. CUDA 驱动未安装或版本不匹配

**强制使用 GPU**（仅在 CUDA 可用时）：
```bash
python train.py trainer.accelerator="gpu" trainer.devices=1
```

**强制使用 CPU**：
```bash
python train.py trainer.accelerator="cpu"
```

### Q: 内存不足怎么办？

- 减少 `batch_size`
- 使用梯度累积
- 启用混合精度训练（在配置中设置）

### Q: 如何查看训练日志？

训练日志默认保存在 `nemo_experiments/` 目录下，或使用 TensorBoard：

```bash
tensorboard --logdir=nemo_experiments
```

### Q: 支持哪些音频格式？

支持常见的音频格式：WAV、MP3、FLAC、OPUS 等。

### Q: 如何自定义模型架构？

修改 `config/nest_fast-conformer.yaml` 中的模型配置，或创建新的配置文件。

更多问题请参考 [INSTALL.md](INSTALL.md) 或查看项目文档。

## 📚 相关文档

- **[INSTALL.md](INSTALL.md)** - 安装和使用指南（包含 Windows 说明）
- **[PROJECT_STRUCTURE_CLEAN.md](PROJECT_STRUCTURE_CLEAN.md)** - 项目结构说明
- **[COMPARISON.md](COMPARISON.md)** - 与 NeMo 的对比分析
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 快速参考
- **[DDP_PERFORMANCE_OPTIMIZATION.md](DDP_PERFORMANCE_OPTIMIZATION.md)** - DDP性能优化指南
- **[PERFORMANCE_COMPARISON_DETAILED.md](PERFORMANCE_COMPARISON_DETAILED.md)** - 详细性能对比和优化指南
- **[DDP_TROUBLESHOOTING.md](DDP_TROUBLESHOOTING.md)** - DDP故障排除指南

## ✅ 与 NeMo 对齐确认

本项目已与 NeMo 原版完全对齐，所有配置参数保持一致：

### 配置对齐 ✅

| 配置项 | NeMo原版 | 本项目 | 状态 |
|--------|---------|--------|------|
| `trainer.strategy` | `auto` | `auto` | ✅ 一致 |
| `trainer.sync_batchnorm` | `true` | `true` | ✅ 一致 |
| `trainer.accelerator` | `auto` | `auto` | ✅ 一致 |
| `train_ds.num_workers` | `0` (默认) | `0` | ✅ 一致 |
| `train_ds.pin_memory` | `true` | `true` | ✅ 一致 |
| DataLoader配置 | 基本配置 | 基本配置 | ✅ 一致 |
| `max_trial` (sample_noise) | `20` | `20` | ✅ 一致 |
| `max_trial` (load_noise_audio) | `100` | `100` | ✅ 一致 |
| `librosa.resample` | 默认 | 默认 | ✅ 一致 |

### 架构对齐 ✅

- ✅ 模型架构与NeMo一致
- ✅ Preprocessor调用逻辑与NeMo一致（双重调用是设计限制）
- ✅ DataLoader创建逻辑与NeMo一致
- ✅ DDP数据分布处理与NeMo一致

### 性能优化 ✅

- ✅ DDP配置已优化（`gradient_as_bucket_view`等选项在示例文件中）
- ✅ 数据加载已优化（与NeMo一致的基本配置）
- ✅ 所有已知性能瓶颈已识别并记录

**注意：** 默认配置与NeMo原版完全一致。如需性能优化，请参考 `nest_fast-conformer_ddp_example.yaml` 和性能优化文档。

## 🤝 贡献

本项目是从 NeMo 框架中提取的简化版本。如需贡献：

1. 确保代码符合项目风格
2. 添加必要的测试
3. 更新相关文档

## 📄 许可证

本项目基于 Apache License 2.0 许可证。详见 LICENSE 文件。

## 🙏 致谢

本项目基于 NVIDIA NeMo 框架开发。感谢 NeMo 团队提供的优秀框架。

## 📧 联系方式

如有问题或建议，请通过 Issue 反馈。

---

**注意**: 本项目已完全独立于 NeMo，可以直接使用。如有问题请查看 [INSTALL.md](INSTALL.md)。
