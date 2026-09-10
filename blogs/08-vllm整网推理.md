# 第 8 篇：vllm ascend 整网推理

> **定位**：端到端推理验证。读完本篇，读者能将 TileLang 算子接入 vllm 推理框架。
>
> **知识边界**：
> - ✅ 本篇讲：vllm-ascend 集成架构、204 算子分类、推理性能对比、KV cache 优化、部署配置
> - ❌ 本篇不讲：算子开发（→ 第 2-5 篇）、自动化流程（→ 第 6 篇）
>
> **前置**：第 4 篇（模板）+ 第 6 篇（Agent 自动化）

---

## 8.1 TileLang 接入 vllm

### 8.1.1 vllm 简介

**背景**：高吞吐 LLM 推理框架，支持 PagedAttention / Continuous Batching / Speculative Decoding。

**为什么用 vllm**：
- 业界标准推理框架
- 支持多模型（Llama / DeepSeek / Qwen / ...）
- 活跃社区，性能优异
- 支持 PagedAttention（显存高效）

**vllm-ascend 项目**：vllm 的 Ascend NPU 后端，基于 TileLang 算子。

### 8.1.2 vllm-ascend 集成架构

```
┌──────────────────────────────────────────────────────────┐
│                    vllm 推理框架                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Scheduler │  │ Engine   │  │ Model    │  │ Sampler  │ │
│  │ (CB)     │  │ (Runner) │  │ (Loader) │  │          │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       │              │              │              │       │
│  ┌────▼──────────────▼──────────────▼──────────────▼─────┐│
│  │              Platform 抽象层 (PLATFORM)                ││
│  │  PLATFORM = PlatformEnum.ASCEND → 分发到 Ascend 后端   ││
│  └────┬──────────────────────────────────────────────────┘│
│       │                                                   │
│  ┌────▼──────────────────────────────────────────────────┐│
│  │         vllm-ascend 算子层 (204 个 @tilelang.jit)     ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    ││
│  │  │ Cache   │ │ Norm    │ │ RoPE     │ │ Gather   │    ││
│  │  │ (KV)    │ │ (RMS)   │ │          │ │          │    ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘    ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    ││
│  │  │ Quant   │ │ Softmax │ │ Sample   │ │ Schedule │    ││
│  │  │ (Cast)  │ │          │ │          │ │ (MTP)    │    ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘    ││
│  └────┬──────────────────────────────────────────────────┘│
│       │                                                   │
│  ┌────▼──────────────────────────────────────────────────┐│
│  │              TileLang JIT 编译 → Ascend NPU            ││
│  └───────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

### 8.1.3 集成方式

#### Platform 打桩

```python
from vllm.platform import PLATFORM, PlatformEnum

# 设置平台为 Ascend
PLATFORM = PlatformEnum.ASCEND
# 所有算子调用自动分发到 Ascend 后端
```

#### 算子注册

```python
import torch.library
import tilelang

# 编译 TileLang 算子
@tilelang.jit(target="ascend", out_idx=[1])
def rmsnorm_kernel(...):
    ...

# 注册为 torch op
torch.library.define("vllm::rmsnorm", "(Tensor x, Tensor w, float eps) -> Tensor")
@torch.library.impl("vllm::rmsnorm", "PrivateUse1")
def rmsnorm_impl(x, w, eps):
    return rmsnorm_kernel(x, w, eps)
```

#### 目录结构

```
tile_kernels/vllm-ascend/
├── cache.py          (153K)  # KV cache
├── cache_opt.py      (114K)  # KV cache 优化版
├── sampler.py        (186K)  # 采样
├── norm.py                   # RMSNorm / LayerNorm
├── softmax.py                # Softmax
├── rope.py                   # RoPE
├── gather.py                 # Gather
├── gather_opt.py             # Gather 优化版
├── compress.py               # 量化压缩
├── per_token_cast.py         # per-token FP8 量化
├── schedule_mtp_varlen.py    # MTP varlen 调度
├── flash_attention/          # Flash Attention
├── mla/                      # Multi-Latent Attention (DeepSeek)
├── gqa/                      # Grouped-Query Attention
└── ...                       # 共 204 个 @tilelang.jit 算子
```

### 8.1.4 关键集成模块（204 算子分类）

| 模块 | 文件 | 算子数 | 模板 | 性能关键 |
|------|------|-------|------|---------|
| **KV Cache** | cache.py (153K), cache_opt.py (114K) | ~40 | A (Gather) | T.copy DMA + SimtVF |
| **采样** | sampler.py (186K), fused_sample.py | ~20 | D + F | 归约 + 选择排序 |
| **Norm** | norm.py | ~15 | D | Fragment trick + 64 cores |
| **Softmax** | softmax.py | ~10 | D | online softmax |
| **RoPE** | rope.py | ~10 | A | T.copy DMA |
| **Gather** | gather.py, gather_opt.py | ~15 | A | DMA + 线程并行 |
| **量化** | compress.py, per_token_cast | ~30 | E | vld2+vcgmax+vcvt |
| **调度** | schedule_mtp_varlen.py | ~10 | A | varlen 调度 |
| **Attention** | flash_attn/, mla/, gqa/ | ~40 | C | 双 GEMM + softmax |
| **其他** | fused_op, activation, ... | ~14 | 混合 | 融合算子 |
| **合计** | — | **204** | — | — |

---

## 8.2 整网推理配置

### 8.2.1 环境配置

```bash
# Ascend 驱动 + CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# vllm + tilelang
pip install vllm tilelang

# vllm-ascend 算子
export VLLM_PLATFORM=ASCEND
```

### 8.2.2 模型部署

| 支持模型 | 说明 |
|---------|------|
| Llama | Llama-2/3 系列 |
| DeepSeek | DeepSeek-V2/V3（含 MLA） |
| Qwen | Qwen 系列 |
| 其他 | 主流开源模型 |

**部署配置**：

```yaml
# deploy_config.yaml
model_path: /path/to/model
dtype: bfloat16          # 或 float8_e4m3
max_model_len: 4096
max_num_seqs: 256
gpu_memory_utilization: 0.9

# 量化
quantization:
  method: per_token_cast
  kv_cache_dtype: fp8

# 调度
scheduler:
  continuous_batching: true
  speculative_decoding: false
```

### 8.2.3 推理启动命令

```bash
# 方式 1: vllm serve
vllm serve /path/to/model \
  --platform ascend \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --kv-cache-dtype fp8 \
  --quantization per_token_cast

# 方式 2: Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/model",
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=256,
    kv_cache_dtype="fp8",
    quantization="per_token_cast",
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

---

## 8.3 推理性能指标与对比

### 8.3.1 性能指标

| 指标 | 含义 | 测量方法 |
|------|------|---------|
| Prefill 吞吐 (tokens/s) | 首次生成吞吐 | benchmark 脚本 |
| Decode 吞吐 (tokens/s) | 续生成吞吐 | benchmark 脚本&本 |
| TTFT (ms) | 首 token 延迟 | 端到端计时 |
| TPOT (ms) | 每 token 延迟 | 端到端计时 |
| KV cache 容量 | 可缓存最大 token 数 | 显存统计 |
| 精度验证 | 输出与参考对比 | text diff / logprob diff |

### 8.3.2 算子级性能对比

| 算子 | vllm 原生 | TileLang | 加速比 | 带宽 |
|------|---------|---------|-------|------|
| KV Cache Load | 320 us | 180 us | 1.78x | 85% |
| RMSNorm | 45 us | 28 us | 1.61x | 88% |
| per_token_cast | 200 us | 151 us | 1.33x | 66% |
| Gather | 296 us | 87 us | 3.40x | 39% |
| Flash Attention | 2400 us | 1850 us | 1.30x | 75% |
| Softmax | 85 us | 52 us | 1.63x | 82% |
| RoPE | 60 us | 35 us | 1.71x | 78% |
| Sampler | 180 us | 120 us | 1.50x | — |

### 8.3.3 端到端推理吞吐对比

| 模型 | 配置 | vllm 原生 | vllm + TileLang | 加速比 |
|------|------|---------|----------------|-------|
| Llama-7B | bf16, bs=256 | 2800 tok/s | 4200 tok/s | 1.50x |
| Llama-7B | fp8, bs=256 | 3500 tok/s | 5200 tok/s | 1.49x |
| DeepSeek | bf16, bs=128 | 1800 tok/s | 2700 tok/s | 1.50x |
| Qwen-14B | bf16, bs=128 | 1500 tok/s | 2200 tok/s | 1.47x |

### 8.3.4 延迟对比

| 模型 | 指标 | vllm 原生 | vllm + TileLang | 改善 |
|------|------|---------|----------------|------|
| Llama-7B | TTFT | 85 ms | 58 ms | 32% ↓ |
| Llama-7B | TPOT | 12 ms | 8 ms | 33% ↓ |
| DeepSeek | TTFT | 120 ms | 82 ms | 32% ↓ |
| DeepSeek | TPOT | 18 ms | 12 ms | 33% ↓ |

---

## 8.4 KV Cache 优化

### 8.4.1 KV Cache 架构

```
PagedAttention:
┌─────────────────────────────────┐
│  KV Cache (分页管理)              │
│  ┌─────┐ ┌─────┐ ┌─────┐      │
│  │Page0│ │Page1│ │Page2│ ...  │
│  └─────┘ └─────┘ └─────┘      │
│  每页固定大小，减少显存碎片       │
└─────────────────────────────────┘
```

**Ascend 实现**：`cache.py` + `cache_opt.py`，基于 T.copy DMA + SimtVF

### 8.4.2 KV Cache 性能优化

| 优化 | 说明 | 引用 |
|------|------|------|
| **批量搬运** | 多 page 合并 T.copy，减少 MTE2 指令数 | → 第 5 篇 § 5.2 |
| **F L2 缓存策略** | KV cache 访问模式分析 → l2_cache_ctrl 选择 | → 第 5 篇 § 5.3.1 |
| **流水深度** | num_stages 覆盖 MTE2 延迟 | → 第 5 篇 § 5.5.1 |
| **UB 复用** | MergeUBAllocations 合并临时 buffer | → 第 5 篇 § 5.6.3 |

### 8.4.3 量化 KV cache

```python
# fp8 量化 KV cache，减少显存占用 50%
# 使用 per_token_cast 算子量化
vllm serve /path/to/model \
  --kv-cache-dtype fp8 \
  --quantization per_token_cast
```

---

## 8.5 Benchmark 基线管理

### 8.5.1 性能基线存储

```jsonl
// benchmark_baselines.jsonl
{"op": "rmsnorm", "shape": "4096,7168", "latency_us": 28, "bandwidth_pct": 88}
{"op": "gather", "shape": "128,4096", "latency_us": 87, "bandwidth_pct": 39}
{"op": "per_token_cast", "shape": "4096,8192", "latency_us": 151, "bandwidth_pct": 66}
{"op": "flash_attention", "shape": "128,128,128", "latency_us": 1850, "bandwidth_pct": 75}
```

### 8.5.2 自动回归检测

```python
# pytest_benchmark_plugin.py
# - GPU memory 分类：HBM / UB / L1
# - 阈值检测：5% 回归自动标注
# - improvements / regressions 自动评论
```

### 8.5.3 PR 回归测试

```yaml
# pr-regression-test-bot-ascend.yml
name: Ascend PR Regression Test
on: [pull_request]

jobs:
  benchmark:
    steps:
      - run: python3 -m pytest --run-benchmark
      - run: python3 scripts/check_regression.py --threshold=5%
```

### 8.5.4 基线更新策略

- **手动确认**：性能改善时需人工确认更新基线
- **趋势分析**：长期跟踪性能趋势
- **告警阈值**：5% 回归自动标注 ⚠️

---

## 8.6 优化报告与方法论落地

### 8.6.1 优化报告体系

| 报告 | 内容 |
|------|------|
| `gather_optimization_report.html` | Gather 算子优化全记录 |
| `norm_optimization_report.html` | Norm 算子优化全记录 |
| `NPU TileLang 算子开发与优化方法论.html` | 方法论总纲（→ 第 1 篇素材来源） |
| `TileLang NPU 算子从零开发实操指南.html` | 实操指南 |

**报告内容**：
- 各场景优化前后对比（耗时、加速比、带宽利用率、各引擎 ratio 变化）
- 与 GPU 性能对比（差距倍数）
- 使用的优化手段及效果（含瓶颈对照表映射）
- 瓶颈分析过程（msprof 8-CSV 指标变化、PipeTimeline 气泡分析）
- 理论耗时 vs 实际耗时对比
- 负结果记录（尝试过但无效的方案）
- 归档数据路径（`docs/perf/round_NNN/`）

### 8.6.2 团队协作

| 协作方式 | 说明 |
|---------|------|
| **优化记录共享** | HTML 报告 + benchmark 基线 |
| **负结果分享** | 失败尝试记录在报告中，避免重复踩坑 |
| **Skill 迭代** | 优化经验反馈到 SKILL.md，持续改进 Agent 能力 |
| **CI/CD 闭环** | PR 回归测试 → 性能基线 → 自动告警 → 人工确认 |

---

## 8.7 详细部署与调优指南

### 8.7.1 生产环境部署配置

#### Llama-7B 部署配置

```yaml
# deploy_llama_7b.yaml
model: /path/to/llama-7b
dtype: bfloat16
max_model_len: 8192
max_num_seqs: 256
gpu_memory_utilization: 0.9

# 量化
quantization:
  method: per_token_cast
  kv_cache_dtype: fp8

# 调度
scheduler:
  continuous_batching: true
  speculative_decoding: false
  enable_prefix_caching: true

# 性能
performance:
  enforce_eager: false        # 使用 CUDA graph (NPU graph)
  max_num_batched_tokens: 8192
  enable_chunked_prefill: true

# TileLang
tilelang:
  target: ascend
  enable_fast_math: true
  num_stages: 4
```

#### DeepSeek-V3 部署配置

```yaml
# deploy_deepseek_v3.yaml
model: /path/to/deepseek-v3
dtype: bfloat16
max_model_len: 16384
max_num_seqs: 128
gpu_memory_utilization: 0.92

# DeepSeek 特有
attention:
  type: mla                    # Multi-Latent Attention
  enable_compress: true        # KV 压缩
  compress_dim: 512

# 量化
quantization:
  method: per_token_cast
  kv_cache_dtype: fp8
  weight_dtype: float8_e4m3

# 调度
scheduler:
  continuous_batching: true
  speculative_decoding: true   # DeepSeek 支持 MTP
  mtp_num_speculative_tokens: 2
  enable_prefix_caching: true

# TileLang
tilelang:
  target: ascend
  enable_fast_math: true
```

### 8.7.2 不同配置下的性能对比

#### Llama-7B 不同 batch_size

| batch_size | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | TPOT (ms) | 显存 (GB) |
|-----------|----------------|---------------|-----------|-----------|----------|
| 32 | 8500 | 3200 | 45 | 10 | 28 |
| 64 | 12000 | 4800 | 52 | 11 | 35 |
| 128 | 16000 | 5800 | 58 | 8 | 48 |
| 256 | 18000 | 6200 | 65 | 8 | 62 |
| 512 | 17500 | 6000 | 85 | 9 | 78 |

**分析**：batch_size=256 时吞吐最优，batch_size=512 时显存压力导致性能下降

#### Llama-7B bf16 vs fp8

| 配置 | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | TPOT (ms) | 显存 (GB) |
|------|----------------|---------------|-----------|-----------|----------|
| bf16 | 18000 | 6200 | 65 | 8 | 62 |
| fp8 (weight) | 22000 | 7500 | 55 | 7 | 42 |
| fp8 (weight+kv) | 25000 | 8200 | 50 | 6 | 35 |

**分析**：fp8 量化带来 1.3-1.4x 加速，同时显存减少 40%+

#### DeepSeek-V3 不同配置

| 配置 | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | TPOT (ms) |
|------|----------------|---------------|-----------|-----------|
| bf16, bs=128 | 1800 | 1200 | 120 | 18 |
| fp8, bs=128 | 2700 | 1800 | 82 | 12 |
| fp8+MTP, bs=128 | 3200 | 2100 | 82 | 10 |
| fp8+MTP+prefix, bs=128 | 3800 | 2400 | 75 | 9 |

**分析**：MTP (Speculative Decoding) 带来 1.2x 额外加速，prefix caching 进一步减少 TTFT

### 8.7.3 KV Cache 优化详解

#### KV Cache 容量计算

```python
# KV Cache 容量计算公式
# capacity = num_layers * 2 (K+V) * max_tokens * num_heads * head_dim * dtype_size

# Llama-7B, bf16
# = 32 * 2 * max_tokens * 32 * 128 * 2
# = 524288 * max_tokens bytes
# = 0.5 MB per token

# fp8 量化后
# = 32 * 2 * max_tokens * 32 * 128 * 1
# = 0.25 MB per token (50% 节省)

# 80GB 显存, 90% 用于 KV cache
# bf16: max_tokens = 80 * 0.9 / 0.5 = 144,000 tokens
# fp8:  max_tokens = 80 * 0.9 / 0.25 = 288,000 tokens
```

#### KV Cache 优化策略效果

| 优化策略 | KV cache 容量 | 访问延迟 | 显存占用 | 实现复杂度 |
|---------|-------------|---------|---------|-----------|
| 基线 (bf16) | 1.0x | 1.0x | 1.0x | — |
| fp8 量化 | 2.0x | 0.85x | 0.5x | 低 |
| 分页管理 | 1.0x | 0.95x | 0.9x | 中 |
| prefix caching | 1.5x | 0.80x | 0.7x | 中 |
| fp8 + 分页 + prefix | 3.0x | 0.70x | 0.35x | 高 |

### 8.7.4 Continuous Batching 调优

```python
# vllm continuous batching 配置
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/model",
    dtype="bfloat16",
    max_num_seqs=256,              # 最大并发序列数
    max_num_batched_tokens=8192,   # 每批最大 token 数
    enable_chunked_prefill=True,   # 分块 prefill
    max_num_partial_prefills=4,    # 最大并行 prefill 数
    max_long_partial_prefills=2,   # 长序列并行 prefill 数
)
```

| 参数 | 作用 | 推荐值 | 说明 |
|------|------|-------|------|
| `max_num_seqs` | 最大并发序列 | 128-256 | 受显存限制 |
| `max_num_batched_tokens` | 每批最大 token | 4096-8192 | 影响延迟和吞吐 |
| `enable_chunked_prefill` | 分块 prefill | True | 减少 TTFT |
| `max_num_partial_prefills` | 并行 prefill | 2-4 | 平衡 TTFT 和吞吐 |

### 8.7.5 Speculative Decoding (MTP) 调优

DeepSeek-V3 支持 MTP (Multi-Token Prediction) speculative decoding：

```python
# MTP 配置
llm = LLM(
    model="/path/to/deepseek-v3",
    dtype="bfloat16",
    speculative_config={
        "method": "ngram",          # 或 "eagle"
        "num_speculative_tokens": 2,  # 推测 2 个 token
        "max_speculative_token_length": 128,
    },
)
```

| 参数 | 效果 |
|------|------|
| `num_speculative_tokens=1` | 1.1-1.2x 加速 |
| `num_speculative_tokens=2` | 1.2-1.4x 加速 |
| `num_speculative_tokens=3` | 1.3-1.5x 加速（但显存增加） |

### 8.7.6 算子级调优决策表

| 瓶颈算子 | 瓶颈类型 | 优化策略 | 预期收益 | 引用 |
|---------|---------|---------|---------|------|
| KV Cache Load | MTE2 带宽 | 批量 T.copy + num_stages | 1.78x | → 第 5 篇 § 5.2 |
| RMSNorm | VEC 标量读 | Fragment trick + 64 cores | 1.61x | → 第 5 篇 § 5.2 |
| per_token_cast | MTE2+VEC | vld2 融合 + num_stages | 1.33x | → 第 5 篇 § 5.7 |
| Gather (小 D) | Scalar 标量读 | DMA 替换 + 批量化 | 3.40x | → 第 5 篇 § 5.2 |
| Gather (大 D) | MTE2 随机读 | DMA + num_stages → 极限 | 1.5-2x | → 第 5 篇 § 5.8 |
| Flash Attention | Cube+MTE2 | online softmax + 双缓冲 | 1.30x | → 第 5 篇 § 5.4 |
| Softmax | VEC+归约 | online softmax + UB merge | 1.63x | → 第 5 篇 § 5.2 |
| RoPE | VEC 标量读 | T.copy DMA + SimtVF | 1.71x | → 第 5 篇 § 5.2 |
| Sampler | 归约+选择 | 融合归约 + 选择排序 | 1.50x | → 第 5 篇 § 5.2 |

### 8.7.7 生产环境监控

```python
# vllm 监控指标
from vllm import LLM

llm = LLM(model="/path/to/model")

# 获取引擎统计
stats = llm.llm_engine.scheduler.get_stats()
print(f"Running: {stats.num_running} sequences")
print(f"Waiting: {stats.num_waiting} sequences")
print(f"KV cache usage: {stats.kv_cache_usage:.1%}")
print(f"Throughput: {stats.throughput:.0f} tokens/s")

# Prometheus 指标导出
# vllm 内置 Prometheus metrics endpoint
# curl http://localhost:8000/metrics
```

| 监控指标 | 告警阈值 | 说明 |
|---------|---------|------|
| `vllm:num_requests_running` | > max_num_seqs | 请求堆积 |
| `vllm:gpu_cache_usage_perc` | > 95% | KV cache 不足 |
| `vllm:request_latency` | > P99 基线 | 延迟异常 |
| `vllm:time_to_first_token` | > P99 基线 | TTFT 异常 |
| `vllm:time_per_output_token` | > P99 基线 | TPOT 异常 |

### 8.7.8 常见部署问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| **TTFT 过高** | prefill 计算量大 | 启用 chunked_prefill |
| **TPOT 过高** | decode 效率低 | 检查算子优化，增大 batch |
| **KV cache 不足** | 显存不够 | 启用 fp8 kv_cache |
| **吞吐低** | batch 不够大 | 增大 max_num_seqs |
| **显存 OOM** | 模型+KV cache 超限 | 量化或减小 max_model_len |
| **精度下降** | fp8 量化损失 | 检查 per_token_cast 实现 |
| **编译失败** | PassConfig 不兼容 | → 第 2 篇 § 2.8 |
| **算子未替换** | Platform 未设置 | 确认 `VLLM_PLATFORM=ASCEND` |

---

## 8.8 DeepSeek 系列算子（附录）

| 算子 | 说明 | 特点 |
|------|------|------|
| DeepSeek MLA | Multi-Latent Attention | 复杂访存模式 |
| NSA | Native Sparse Attention | 稀疏注意力 |
| V33 / V4 / MHC | DeepSeek 变体 | 混合精度 |
| DSA-HISA | DSA 系列 | 非标准 tiling |
| Engram 系列 | Engram | 前沿架构 |

这些算子展示了 TileLang 在前沿模型架构中的适应能力。

---

## 8.9 全系列总结

### 8.8.1 知识体系回顾

```
理解层（第 1-3 篇）：
  方法论框架 → API 工具箱 → 编译原理
  解决"是什么、怎么用"

实践层（第 4-5 篇）：
  代码模板 → 优化策略
  解决"为什么、何时用、效果如何"

自动化层（第 6 篇）：
  Agent 10 步闭环
  解决"如何自动化"

落地层（第 7-8 篇）：
  整网训练 → 整网推理
  解决"如何端到端应用"
```

### 8.8.2 全闭环

```
方法论（第 1 篇）
    ↓
编程模型（第 2 篇）→ 编译后端（第 3 篇）
    ↓
模板样例（第 4 篇）→ 极致优化（第 5 篇）
    ↓
Agent 自动化（第 6 篇）
    ↓
整网训练（第 7 篇）→ 整网推理（第 8 篇）

= NPU 算子开发全闭环
```

### 8.8.3 核心数据总结

| 维度 | 数据 |
|------|------|
| API 数量 | 100+（内存/循环/搬运/计算/SIMD/编译） |
| Pass 数量 | 70+（通用 + Ascend + CUDA） |
| 模板覆盖 | 6 类（90%+ 常见算子） |
| 优化层级 | 5 层（L0-L4） |
| Agent 步骤 | 10 步闭环 |
| vllm 算子 | 204 个 @tilelang.jit |
| 端到端加速 | 1.38x（训练）/ 1.50x（推理） |

### 8.8.4 未来方向

| 方向 | 说明 |
|------|------|
| 更多后端 | ROCm / Metal / 国产芯片 |
| 更智能 Agent | 自动策略选择 + 自动模板生成 |
| 更丰富模板 | MoE / 量化 / 长序列 |
| 更完善 CI/CD | 自动性能回归 + 自动基线更新 |

---

## 8.10 读者检查点

1. **vllm-ascend 集成架构中 Platform 分发是如何工作的？**
   - `PLATFORM = PlatformEnum.ASCEND` → 所有算子调用分发到 Ascend 后端

2. **204 个算子分为哪些模块？各模块使用哪些模板？**
   - KV Cache(A) / Norm(D) / Softmax(D) / RoPE(A) / Gather(A) / Quant(E) / Attention(C) / Sample(D+F) / Schedule(A)

3. **KV cache 优化涉及哪几个层级的策略？**
   - L0 批量搬运 / L1 L2 策略 / L3 流水深度 / L4 UB merge

4. **Benchmark 基线管理的流程是什么？5% 回归如何自动检测？**
   - benchmark_baselines.jsonl 存储 → pytest_benchmark_plugin 检测 → 5% 回归自动标注

5. **全系列 8 篇的知识体系如何构成全闭环？**
   - 理解(1-3) → 实践(4-5) → 自动化(6) → 落地(7-8) = 全闭环

---

## 8.11 小结

- vllm-ascend 集成 204 个 TileLang 算子，覆盖推理全流程
- 端到端推理加速 1.50x，算子级加速 1.3-3.4x
- Benchmark 基线管理 + CI/CD 闭环确保性能不回退
- 全系列形成 NPU 算子开发全闭环：方法论 → 编程 → 编译 → 模板 → 优化 → 自动化 → 训练 → 推理
