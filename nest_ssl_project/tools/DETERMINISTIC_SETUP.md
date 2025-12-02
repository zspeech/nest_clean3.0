# 确定性设置说明

## 已完成的随机性控制

### 1. 禁用 Preprocessor 的 Dither
**修改文件**:
- `nest_ssl_project/config/nest_fast-conformer.yaml`: 将 `dither: 0.00001` 改为 `dither: 0.0`
- `nemo.bat`: 添加 `model.preprocessor.dither=0.0`
- `nest_train.bat`: 添加 `model.preprocessor.dither=0.0`

**说明**: `dither` 参数会在音频处理时添加白噪声，导致每次运行结果不同。禁用后可以确保预处理结果完全一致。

### 2. 已存在的确定性设置
- ✅ `model.train_ds.seed=42`: 数据加载随机种子
- ✅ `seed=42`: 全局随机种子
- ✅ `model.train_ds.batch_augmentor.prob=0.0`: 禁用批量增强
- ✅ `model.train_ds.shuffle=false`: 禁用数据打乱
- ✅ 所有 dropout 设置为 0.0

### 3. 噪声采样
当 `noise_manifest` 为 `null` 时，`sample_noise` 函数会直接返回全零的噪声张量，不涉及随机数生成。

## 验证步骤

1. **重新运行训练**:
   ```bash
   # NeMo
   ./nemo.bat
   
   # nest_ssl_project
   ./nest_train.bat
   ```

2. **比较结果**:
   ```bash
   ./compare_outputs.bat
   ```

3. **检查输入数据差异**:
   - 如果输入数据仍然有差异（> 1e-6），可能需要检查：
     - 音频文件加载的一致性
     - 浮点数精度设置
     - PyTorch 的确定性模式

## 如果输入数据仍有差异

如果禁用 dither 后输入数据仍有微小差异，可能的原因：

1. **浮点数精度累积**: 不同的计算顺序可能导致微小的数值差异
2. **音频文件加载**: 不同的音频库或版本可能产生微小差异
3. **PyTorch 确定性模式**: 可以尝试启用 `torch.use_deterministic_algorithms(True)`

## 下一步

如果输入数据差异仍然存在，建议：
1. 检查 `featurizer.process` 的实现是否完全一致
2. 对比 NeMo 和 nest_ssl_project 的音频加载逻辑
3. 考虑使用完全相同的输入数据（从文件加载，而不是实时生成）

