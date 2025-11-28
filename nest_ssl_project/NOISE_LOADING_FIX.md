# 噪声加载错误处理和卡住问题修复

## 🐛 问题诊断

### 问题1: 训练到71个batch卡住
**根本原因：** `load_noise_audio`函数中的while循环没有异常处理

**问题代码：**
```python
while cnt < max_trial:
    audio_segment = AudioSegment.from_file(...)  # 如果这里抛出异常，cnt不会递增
    if audio_segment.samples.numel() > 0:
        break
    cnt += 1  # 只有成功加载但为空时才会执行
```

**问题分析：**
- 如果`AudioSegment.from_file`抛出异常（文件损坏、不存在等），while循环会一直重试
- `cnt`不会递增，导致无限循环
- 训练进程卡住，无法继续

### 问题2: 日志打印不一致
**问题：** 使用了`is_global_rank_zero()`检查，与NeMo不一致

**NeMo的行为：** 所有rank都打印warning日志
**我们的行为：** 只有rank 0打印warning日志

---

## ✅ 修复方案

### 1. 添加异常处理到while循环

**修复前：**
```python
while cnt < max_trial:
    audio_segment = AudioSegment.from_file(...)  # 可能抛出异常
    if audio_segment.samples.numel() > 0:
        break
    cnt += 1
```

**修复后：**
```python
cnt = 0
audio_segment = None
while cnt < max_trial:
    try:
        offset = np.random.uniform(0, duration - max_dur)
        audio_segment = AudioSegment.from_file(...)
        if audio_segment.samples.numel() > 0:
            break
    except Exception as e:
        # 如果加载失败，递增计数器并重试
        # 异常会被sample_noise捕获
        pass
    cnt += 1  # 无论成功还是失败都递增

# 如果所有重试都失败，抛出异常给sample_noise处理
if audio_segment is None:
    raise RuntimeError(f"Failed to load noise audio after {max_trial} attempts")
```

**关键改进：**
- ✅ `cnt`在`except`块外递增，确保每次循环都递增
- ✅ 如果所有重试失败，抛出异常给`sample_noise`处理
- ✅ 避免无限循环

### 2. 对齐日志打印（与NeMo一致）

**修复前：**
```python
if is_global_rank_zero():
    logging.warning("Error loading noise audio...")
```

**修复后：**
```python
logging.warning("Error loading noise audio...")  # 所有rank都打印，与NeMo一致
```

**原因：** NeMo的实现中，所有rank都打印warning，这是为了调试和监控。我们的实现应该保持一致。

---

## 🔍 错误处理流程（与NeMo对齐）

### NeMo的错误处理流程：

1. **`load_noise_audio`**: 
   - 如果文件加载失败，while循环会重试（但NeMo的实现也没有异常处理，可能也有同样的问题）
   - 如果所有重试失败，返回空音频或抛出异常

2. **`sample_noise`**:
   - 捕获`load_noise_audio`的异常
   - 打印warning并重试（最多`max_trial`次）
   - 如果所有重试失败，返回zero noise

### 我们的错误处理流程（修复后）：

1. **`load_noise_audio`**:
   - ✅ while循环中添加异常处理
   - ✅ 如果所有重试失败，抛出`RuntimeError`
   - ✅ 确保`cnt`总是递增，避免无限循环

2. **`sample_noise`**:
   - ✅ 捕获`load_noise_audio`的异常
   - ✅ 打印warning并重试（与NeMo一致，所有rank都打印）
   - ✅ 如果所有重试失败，返回zero noise

---

## 📊 修复效果

### 修复前：
- ❌ 训练可能在某个batch卡住（无限循环）
- ❌ 只有rank 0打印日志（与NeMo不一致）

### 修复后：
- ✅ 不会卡住（异常被正确处理）
- ✅ 所有rank都打印日志（与NeMo一致）
- ✅ 错误处理逻辑与NeMo对齐

---

## 🧪 测试建议

1. **测试损坏的噪声文件：**
   - 创建一个损坏的噪声文件
   - 确保训练不会卡住，而是跳过并继续

2. **测试不存在的噪声文件：**
   - 在manifest中引用不存在的文件
   - 确保训练不会卡住，而是返回zero noise

3. **测试DDP训练：**
   - 确保所有rank都能正确处理错误
   - 确保日志输出与NeMo一致

---

## 📝 代码变更总结

**文件：** `nest_ssl_project/data/ssl_dataset.py`

1. **`load_noise_audio`函数**:
   - 添加try-except到while循环
   - 确保`cnt`总是递增
   - 如果所有重试失败，抛出异常

2. **日志打印**:
   - 移除`is_global_rank_zero()`检查
   - 与NeMo对齐，所有rank都打印

---

**更新日期**: 2025-01-XX  
**版本**: 1.0

