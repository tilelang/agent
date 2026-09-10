# 第 7 篇：torchtitan 整网训练

> **定位**：端到端训练验证。读完本篇，读者能将 TileLang 算子接入训练框架。
>
> **知识边界**：
> - ✅ 本篇讲：集成架构、算子替换清单、训练配置、性能对比、问题排查
> - ❌ 本篇不讲：算子开发（→ 第 2-5 篇）、自动化流程（→ 第 6 篇）
>
> **前置**：第 4 篇（模板）+ 第 6 篇（Agent 自动化）

---

## 7.1 TileLang 接入 torchtitan

### 7.1.1 torchtitan 简介

**背景**：PyTorch 官方 Titan 训练框架，支持 Llama 等大模型训练。

**架构**：

```
┌─────────────────────────────────────────────────┐
│                 torchtitan 训练框架               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │  模型定义  │  │  优化器    │  │  数据加载  │   │
│  │  (Llama)  │  │ (AdamW)   │  │ (DataLoader)│  │
│  └─────┬─────┘  └───────────┘  └───────────┘   │
│        │                                         │
│  ┌─────▼─────────────────────────────────┐      │
│  │       torch 原生算子 ← 替换为 TileLang    │      │
│  │  RMSNorm │ RoPE │ Attention │ Linear  │      │
│  └─────┬─────────────────────────────────┘      │
│        │                                         │
│  ┌─────▼─────────────────────────────────┐      │
│  │       TileLang JIT 编译算子              │      │
│  │  @tilelang.jit(target="ascend")       │      │
│  └─────┬─────────────────────────────────┘      │
│        │                                         │
│  ┌─────▼─────────────────────────────────┐      │
│  │       Ascend NPU 硬件执行               │      │
│  └───────────────────────────────────────┘      │
└─────────────────────────────────────────────────┘
```

**为什么选 torchtitan**：
- 官方维护，结构清晰
- 易于替换算子（monkey patch）
- 支持 FSDP / Tensor Parallel / Pipeline Parallel
- 支持混合精度训练（bf16 + fp32 主权重）

### 7.1.2 接入步骤

```
1. 环境准备 → 安装 tilelang + Ascend + torch_npu
2. 识别可替换算子 → 分析模型计算图，标记热点算子
3. 编写 TileLang 算子 → 基于第 4 篇模板 + 第 5 篇优化
4. 正确性验证 → 逐算子 torch.allclose
5. 注册为 torch 自定义 op → torch.library.define + impl
6. 模型中替换原生算子 → monkey patch
7. 端到端训练验证 → loss 收敛 + 吞吐对比
```

### 7.1.3 算子替换清单

| 原生算子 | TileLang 替换 | 模板 | 性能关键 | 开发方式 |
|---------|-------------|------|---------|---------|
| `torch.nn.RMSNorm` | `rmsnorm_asc` | D | Fragment trick + 64 cores | Agent 自动化 |
| `RoPE` | `rope_asc` | A | T.copy DMA + SimtVF | Agent 自动化 |
| `Attention (SDPA)` | `flash_attn_asc` | C | 双 GEMM + online softmax | 手工 + Agent |
| `Linear (F.linear)` | `gemm_asc` | B | TILE_K + aswt_swizzle | 手工 + Agent |
| `SwiGLU` | `swiglu_asc` | D + B 融合 | 归约 + GEMM 融合 | 手工 |
| `Cross Entropy` | `ce_asc` | D | 归约 + log_softmax | Agent 自动化 |

### 7.1.4 注册为 torch 自定义 op

```python
import torch
import torch.library

# 定义 op schema
torch.library.define("tilelang::rmsnorm", "(Tensor x, Tensor w, float eps) -> Tensor")

# 实现
@torch.library.impl("tilelang::rmsnorm", "PrivateUse1")  # NPU device
def rmsnorm_impl(x, w, eps):
    # 调用 TileLang 编译的 kernel
    return rmsnorm_kernel(x, w, eps)

# autograd 支持
def rmsnorm_backward(grad_output, x, w, eps):
    # 需要实现 backward
    ...

torch.library.define("tilelang::rmsnorm_backward",
    "(Tensor grad_output, Tensor x, Tensor w, float eps) -> (Tensor, Tensor)")
@torch.library.impl("tilelang::rmsnorm_backward", "PrivateUse1")
def rmsnorm_backward_impl(grad_output, x, w, eps):
    return rmsnorm_backward_kernel(grad_output, x, w, eps)
```

### 7.1.5 模型中替换

```python
# monkey patch 方式
import torch.nn as nn

class TileLangRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return torch.ops.tilelang.rmsnorm(x, self.weight, self.eps)

# 替换模型中的 RMSNorm
model = LlamaForCausalLM(config)
for name, module in model.named_modules():
    if isinstance(module, nn.RMSNorm):
        setattr(model, name, TileLangRMSNorm(module.weight.shape[0], module.eps))
```

---

## 7.2 整网训练配置

### 7.2.1 环境配置

```bash
# Ascend 驱动 + CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# conda 环境
conda activate tilelang

# torch_npu 适配
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

### 7.2.2 模型选择与配置

| 支持模型 | 参数量 | 说明 |
|---------|-------|------|
| Llama-3B | 3B | 测试用 |
| Llama-7B | 7B | 常用 |
| Llama-13B | 13B | 大规模 |

**训练配置**：

```yaml
# train_config.yaml
model: llama-7b
batch_size: 32
seq_len: 4096
micro_batch_size: 4

# 并行策略
parallel:
  fsdp: true
  tensor_parallel: 2
  pipeline_parallel: 1

# 混合精度
mixed_precision: bf16
gradient_checkpointing: true

# 算子替换
use_tilelang_ops: true
tilelang_target: ascend
```

### 7.2.3 训练启动命令

```bash
torchrun --nproc_per_node=8 train.py \
  --model llama-7b \
  --use-tilelang-ops \
  --tilelang-target ascend \
  --batch-size 32 \
  --seq-len 4096 \
  --mixed-precision bf16 \
  --fsdp \
  --gradient-checkpointing
```

---

## 7.3 训练指标与性能对比

### 7.3.1 性能指标

| 指标 | 含义 | 测量方法 |
|------|------|---------|
| 吞吐量 (tokens/s) | 每秒训练 token 数 | 训练日志统计 |
| Loss 收敛 | 训练 loss 曲线 | 与原生实现对比 |
| 算子级延迟 | 逐算子 msprof 分析 | msprof 采集 |
| 峰值显存 | 训练过程最大显存 | `torch.cuda.max_memory_allocated` |
| 数值精度 | 输出 allclose | `torch.allclose(rtol=1e-3, atol=1e-3)` |

### 7.3.2 算子级性能对比

| 算子 | torch 原生延迟 | TileLang 延迟 | 加速比 | 带宽利用率 |
|------|-------------|-------------|-------|---------|
| RMSNorm | 85 us | 62 us | 1.37x | 88% |
| RoPE | 120 us | 78 us | 1.54x | 72% |
| Attention | 2400 us | 1850 us | 1.30x | 75% |
| Linear | 1800 us | 1450 us | 1.24x | 80% |
| SwiGLU | 320 us | 210 us | 1.52x | 82% |

### 7.3.3 端到端吞吐对比

| 配置 | torch 原生 | TileLang 替换 | 加速比 |
|------|---------|-------------|-------|
| Llama-7B, bs=32, seq=4096 | 4200 tokens/s | 5800 tokens/s | 1.38x |
| Llama-13B, bs=16, seq=4096 | 2100 tokens/s | 2900 tokens/s | 1.38x |
| Llama-7B, bs=64, seq=2048 | 8500 tokens/s | 11500 tokens/s | 1.35x |

### 7.3.4 Loss 收敛对比

```
Step    torch 原生    TileLang    差异
100     6.82         6.83        0.01
500     5.45         5.46        0.01
1000    4.72         4.73        0.01
2000    4.15         4.16        0.01
→ Loss 曲线一致，数值精度无影响
```

---

## 7.4 常见问题与排查

### 7.4.1 问题排查流程图

```
训练异常
  ├─ Loss 不收敛 → 检查算子数值精度 (allclose)
  │   ├─ 精度不匹配 → 检查 cast/round 模式
  │   └─ 精度匹配 → 检查 backward 算子
  ├─ 训练速度下降 → msprof 逐算子分析
  │   ├─ 算子本身慢 → → 第 5 篇优化
  │   └─ launch 开销大 → 融合算子减少 launch
  ├─ 显存增加 → 检查中间 buffer
  │   ├─ UB 分配过多 → → 第 5 篇 § 5.6.3 UB merge
  │   └─ gradient checkpointing 未启用
  └─ 反向传播报错 → 检查 backward 实现
      ├─ backward 缺失 → 同时实现 forward + backward
      └─ autograd 不兼容 → torch.autograd.Function
```

### 7.4.2 常见问题清单

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| Loss 不收敛 | 算子数值精度不匹配 | 检查 cast/round 模式 |
| 训练速度下降 | launch 开销大 | 融合算子减少 kernel launch |
| 显存增加 | 中间 buffer 分配多 | UB merge / gradient checkpointing |
| 反向传播报错 | backward 缺失 | 实现 forward + backward |
| 分布式不兼容 | FSDP/TP 梯度同步 | 检查自定义算子梯度同步 |
| 编译失败 | PassConfig 不兼容 | → 第 2 篇 § 2.8 |

### 7.4.3 数值精度验证

```python
# 逐算子验证
for name, tilelang_op, native_op in ops_to_test:
    x = torch.randn(shape, device="npu")
    out_tilelang = tilelang_op(x)
    out_native = native_op(x)
    assert torch.allclose(out_tilelang, out_native, rtol=1e-3, atol=1e-3), \
        f"{name}: mismatch"
    print(f"{name}: ✅ allclose pass")
```

---

## 7.5 详细环境搭建与配置

### 7.5.1 完整环境搭建步骤

```bash
# 1. Ascend 驱动安装（root 权限）
sudo ./Ascend-HDK-*.run
sudo ./Ascend-driver-*.run
sudo ./Ascend-firmware-*.run

# 2. CANN 工具包安装
sudo ./Ascend-cann-toolkit_*.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. Python 环境创建
conda create -n tilelang python=3.10
conda activate tilelang

# 4. 安装 PyTorch + torch_npu
pip install torch==2.4.0
pip install torch-npu==2.4.0

# 5. 安装 TileLang
cd /path/to/tilelang
pip install -e .

# 6. 安装 torchtitan
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
pip install -e .

# 7. 验证安装
python -c "import tilelang; print(tilelang.__version__)"
python -c "import torch_npu; print(torch_npu.npu.is_available())"
```

### 7.5.2 常见环境问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `No module named 'torch_npu'` | torch_npu 未安装 | `pip install torch-npu` |
| `NPU device not available` | 驱动未正确安装 | 重新安装 Ascend 驱动 |
| `CANN version mismatch` | CANN 与 torch_npu 版本不匹配 | 查阅版本兼容矩阵 |
| `Compilation failed` | CANN 环境变量未设置 | `source set_env.sh` |
| `Out of memory` | 显存不足 | 减小 batch_size 或启用 gradient_checkpointing |

### 7.5.3 不同模型规模的训练配置

#### Llama-3B（测试/调试用）

```yaml
# config_llama_3b.yaml
model:
  name: llama-3b
  num_layers: 28
  num_heads: 16
  dim: 3072
  intermediate_size: 8192

training:
  batch_size: 8
  seq_len: 2048
  learning_rate: 3.0e-4
  warmup_steps: 200

parallel:
  fsdp: true
  tensor_parallel: 1
  pipeline_parallel: 1

mixed_precision: bf16
gradient_checkpointing: false
use_tilelang_ops: true
```

#### Llama-7B（常用训练）

```yaml
# config_llama_7b.yaml
model:
  name: llama-7b
  num_layers: 32
  num_heads: 32
  dim: 4096
  intermediate_size: 11008

training:
  batch_size: 32
  seq_len: 4096
  learning_rate: 3.0e-4
  warmup_steps: 500
  micro_batch_size: 4  # gradient accumulation

parallel:
  fsdp: true
  tensor_parallel: 2
  pipeline_parallel: 1

mixed_precision: bf16
gradient_checkpointing: true
use_tilelang_ops: true
tilelang_target: ascend
```

#### Llama-13B（大规模训练）

```yaml
# config_llama_13b.yaml
model:
  name: llama-13b
  num_layers: 40
  num_heads: 40
  dim: 5120
  intermediate_size: 13824

training:
  batch_size: 16
  seq_len: 4096
  learning_rate: 2.0e-4
  warmup_steps: 1000
  micro_batch_size: 2

parallel:
  fsdp: true
  tensor_parallel: 4
  pipeline_parallel: 2

mixed_precision: bf16
gradient_checkpointing: true
use_tilelang_ops: true
tilelang_target: ascend
```

### 7.5.4 并行策略选择指南

| 模型规模 | GPU/NPU 数 | 推荐策略 | 说明 |
|---------|-----------|---------|------|
| ≤ 3B | 1-2 | FSDP | 简单高效 |
| 7B | 4-8 | FSDP + TP=2 | FSDP 为主，TP 减少通信 |
| 13B | 8-16 | FSDP + TP=4 | TP 减少 AllReduce 开销 |
| 70B | 32+ | FSDP + TP=8 + PP=2 | 三维并行 |

---

## 7.6 Backward 算子实现详解

### 7.6.1 为什么需要 Backward

训练需要反向传播梯度。每个 TileLang 自定义算子必须同时实现 forward 和 backward：

```python
# torch.autograd.Function 方式
class TileLangRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # 保存反向传播需要的中间值
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        x_normed = (x.float() / rms).to(x.dtype)
        ctx.save_for_backward(x_normed, weight, rms)
        return x_normed * weight

    @staticmethod
    def backward(ctx, grad_output):
        x_normed, weight, rms = ctx.saved_tensors
        # 需要实现 backward kernel
        grad_x, grad_w = rmsnorm_backward_kernel(grad_output, x_normed, weight, rms)
        return grad_x, grad_w, None  # eps 不需要梯度
```

### 7.6.2 各算子 Backward 实现复杂度

| 算子 | Forward 模板 | Backward 模板 | Backward 复杂度 | 说明 |
|------|------------|-------------|---------------|------|
| RMSNorm | D | D | 中 | 需计算 mean(x²) 和 x·grad_output |
| RoPE | A | A | 低 | 旋转的逆变换 |
| Attention | C | C | 高 | 需要 dQ/dK/dV 三个梯度 |
| Linear | B | B | 低 | 矩阵转置乘法 |
| SwiGLU | D+B | D+B | 中 | 需 sigmoid 导数 |
| Cross Entropy | D | D | 中 | softmax + one_hot |

### 7.6.3 Backward 算子开发建议

1. **先验证 forward 正确**：确保 forward allclose 通过后再开发 backward
2. **使用 torch.autograd.gradcheck**：数值梯度验证
3. **复用 forward 的 tiling 策略**：backward 通常与 forward 计算模式相似
4. **注意梯度精度**：backward 中间计算用 float32，避免 bf16 精度损失

```python
# 梯度数值验证
from torch.autograd import gradcheck

# 创建测试输入
x = torch.randn(4, 8, dtype=torch.double, device="npu", requires_grad=True)
w = torch.randn(8, dtype=torch.double, device="npu", requires_grad=True)

# 数值梯度检查
assert gradcheck(TileLangRMSNorm.apply, (x, w, 1e-6), eps=1e-6, atol=1e-4)
print("梯度验证通过 ✅")
```

---

## 7.7 训练性能 Profiling

### 7.7.1 算子级 Profiling

```python
# 使用 msprof 对训练过程采集
import torch_npu

# 标记 profiling 范围
with torch_npu.profile(
    output_dir="./train_profile",
    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
    record_shapes=True,
    with_stack=True
) as prof:
    for step in range(10):
        loss = model(batch)
        loss.backward()
        optimizer.step()
        prof.step()

# 导出 Chrome Trace
prof.export_chrome_trace("./train_trace.json")
```

### 7.7.2 热点算子识别

```python
# 分析 profiling 结果
events = prof.key_averages()
sorted_events = sorted(events, key=lambda e: e.npu_time_total, reverse=True)

print("Top 10 热点算子:")
for evt in sorted_events[:10]:
    print(f"  {evt.key}: {evt.npu_time_total/1e3:.1f} ms "
          f"({evt.count} calls, avg {evt.npu_time_total/evt.count/1e3:.1f} us)")
```

### 7.7.3 训练吞吐优化决策树

```
训练吞吐低
  ├─ 算子级延迟高？
  │   ├─ 是 → msprof 逐算子分析 → 第 5 篇优化
  │   └─ 否 → 检查通信和 launch 开销
  │
  ├─ 通信开销大？
  │   ├─ AllReduce 慢 → 增大 TP 减少通信量
  │   ├─ AllGather 慢 → 调整 FSDP shard 大小
  │   └─ Pipeline 气泡 → 调整 micro_batch_size
  │
  ├─ Launch 开销大？
  │   ├─ kernel 数量多 → 融合算子
  │   └─ 小算子多 → 增大 batch_size
  │
  └─ 显存利用率低？
      ├─ 未启用 gradient_checkpointing → 启用
      └─ batch_size 太小 → 增大 batch_size
```

---

## 7.8 实战调试案例

### 7.8.1 案例：Loss 不收敛排查

**现象**：Llama-7B 训练，替换 RMSNorm 后 loss 不收敛

**排查过程**：

```
1. 检查 forward 正确性
   → torch.allclose(tilelang_rmsnorm(x), native_rmsnorm(x), rtol=1e-3)
   → ✅ 通过

2. 检查 backward 正确性
   → gradcheck(TileLangRMSNorm.apply, (x, w, eps))
   → ❌ 失败！梯度不匹配

3. 分析 backward kernel
   → 发现中间计算用 bf16，精度损失
   → 修改：中间计算用 float32

4. 重新验证
   → gradcheck ✅
   → loss 收敛 ✅
```

**根因**：backward 中 `mean(x²)` 在 bf16 下精度不足，需用 float32 中间计算

### 7.8.2 案例：训练速度下降排查

**现象**：替换 Attention 后训练速度下降 20%

**排查过程**：

```
1. msprof 采集训练 profile
   → 发现 Attention 算子延迟正常（1850us）
   → 但 launch 次数是原来的 3x

2. 分析原因
   → TileLang Attention 拆分为 QK^T 和 softmax·V 两个 kernel
   → 原生 SDPA 融合为一个 kernel

3. 优化方案
   → 融合 QK^T + softmax + ·V 为一个 kernel
   → 或使用 torch.compile 融合

4. 验证
   → launch 次数减少 2x
   → 训练速度恢复 ✅
```

**根因**：算子拆分导致 kernel launch 开销增加

### 7.8.3 案例：显存增加排查

**现象**：替换 SwiGLU 后显存增加 30%

**排查过程**：

```
1. 分析显存
   → torch.cuda.memory_summary()
   → 发现中间 buffer 分配过多

2. 分析原因
   → SwiGLU backward 需要保存 sigmoid(x) 中间值
   → 每个 layer 多存一个 [batch, seq, dim] buffer

3. 优化方案
   → 方案 A: gradient_checkpointing 重算 sigmoid
   → 方案 B: 融合 sigmoid 到 forward 输出中

4. 选择方案 B
   → 修改 forward 输出 (output, sigmoid_x)
   → backward 从 ctx 取 sigmoid_x

5. 验证
   → 显存恢复正常 ✅
```

**根因**：backward 中间值保存策略不当

---

## 7.9 读者检查点

1. **TileLang 算子接入 torchtitan 的 7 个步骤是什么？**
   - 环境准备 → 识别算子 → 编写算子 → 正确性验证 → 注册 op → 替换原生 → 端到端验证

2. **哪些算子适合用 Agent 自动化开发？哪些需要手工？**
   - Agent：RMSNorm/RoPE/CE（模板 D/A，规则计算）
   - 手工：Attention/SwiGLU（模板 C/B+D，复杂协作）

3. **算子替换后 loss 不收敛，排查流程是什么？**
   - allclose 检查 → cast/round 模式 → backward 算子

4. **如何测量端到端训练吞吐量？需要对比哪些指标？**
   - tokens/s + Loss 曲线 + 算子级延迟 + 显存 + 精度

5. **反向传播算子缺失时有哪些解决方案？**
   - 实现 backward + torch.autograd.Function

---

## 7.10 小结

- 整网训练验证了 TileLang 算子的数值正确性和端到端性能
- 算子替换清单覆盖模型主要热点算子
- 性能提升 1.3-1.5x（算子级），端到端 1.38x（训练吞吐）
- 下一篇将展示推理场景的应用
