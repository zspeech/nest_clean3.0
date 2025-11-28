# DDP 策略故障排除指南

## 问题：find_unused_parameters 参数不存在

### 问题描述
在 PyTorch Lightning 2.0+ 中，`DDPStrategy` 不再支持 `find_unused_parameters` 参数。如果配置文件中包含此参数，会出现以下错误：
```
TypeError: __init__() got an unexpected keyword argument 'find_unused_parameters'
```

### 解决方案

#### 1. 移除 find_unused_parameters 参数
从配置文件中移除 `find_unused_parameters` 参数：

**错误配置：**
```yaml
strategy:
  _target_: lightning.pytorch.strategies.DDPStrategy
  find_unused_parameters: true  # ❌ 这个参数在 PL 2.0+ 中不存在
  gradient_as_bucket_view: true
```

**正确配置：**
```yaml
strategy:
  _target_: lightning.pytorch.strategies.DDPStrategy
  gradient_as_bucket_view: true
  static_graph: false
```

#### 2. 处理未使用参数错误
如果遇到以下错误：
```
RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss...
```

**解决方法：**

1. **确保所有参数都被使用**
   - 检查模型的前向传播，确保所有参数都参与计算
   - 避免在条件分支中跳过某些层

2. **使用 static_graph 优化**
   ```yaml
   strategy:
     _target_: lightning.pytorch.strategies.DDPStrategy
     static_graph: true  # 如果模型结构不变，设置为 true 可以提高性能
   ```

3. **检查模型实现**
   - 确保所有模型参数都在 `forward()` 方法中被使用
   - 避免使用 `torch.no_grad()` 包裹整个模型

#### 3. 推荐的 DDP 配置

**基本配置（推荐）：**
```yaml
trainer:
  strategy: "ddp"  # 简单字符串格式，自动处理大部分情况
  sync_batchnorm: true
```

**高级配置（需要自定义时）：**
```yaml
trainer:
  strategy:
    _target_: lightning.pytorch.strategies.DDPStrategy
    gradient_as_bucket_view: true  # 内存优化
    static_graph: false  # 如果模型结构不变，设为 true
  sync_batchnorm: true
```

### PyTorch Lightning 版本兼容性

| Lightning 版本 | find_unused_parameters 支持 |
|---------------|---------------------------|
| < 2.0         | ✅ 支持                    |
| >= 2.0        | ❌ 已移除                  |

### 相关资源
- [PyTorch Lightning DDPStrategy 文档](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html)
- [PyTorch DDP 文档](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

