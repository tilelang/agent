以下是使用 `tilelang-operator-dev` skill 从 CPU 实现自动开发高性能 NPU 算子的完整流程：

## 整体流程

```text
CPU 实现（参考） → Step ① 分析规格 → Step ② 选模板生成 NPU kernel
→ Step ③ 正确性验证(对照 CPU) → Step ④⑤ msprof 采集
→ Step ⑦ 瓶颈诊断 → Step ⑧ 优化 → Step ⑨ 验证 → 迭代至达标
```

## 详细操作步骤

### 1. 准备输入

将 CPU 实现作为参考实现，整理算子规格：

```python
# cpu_reference.py — CPU 参考实现
import torch

def my_op_cpu(x, w, eps=1e-6):
    # 例如 RMSNorm
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms).to(x.dtype) * w
```

### 2. 触发 skill，执行 10 步闭环

向 Agent 提供以下输入，skill 会自动驱动 10 步闭环：

```markdown
# 算子优化任务

## 算子规格
- 名称：my_op
- 语义：RMSNorm，y = x / sqrt(mean(x²) + eps) * w
- shape：x [4096, 7168], w [7168]
- dtype：float32
- 访问模式：归约 + elementwise

## 参考实现
- CPU 实现：cpu_reference.py:my_op_cpu
- 关键维度：D=7168

## 约束
- 每次修改后验证正确性（对照 CPU 实现）
- 每次只改一个主要变量
- 记录所有尝试过的方案（含负结果）
- 分析交付使用中文
```

### 3. Skill 自动执行的 10 步

| 步骤             | Skill 动作                                               | 使用 skill 的哪部分内容                   |
| ---------------- | -------------------------------------------------------- | ----------------------------------------- |
| ① 分析golden     | 识别为“归约/Norm”→ 模板 D                                | §3.0 决策树 + §3.4 模板 D                 |
| ② 编写初始版本   | 从 `references/templates/template_d_rmsnorm.py` 生成代码 | §2 API 速查 + §3.4 模板                   |
| ③ 正确性验证     | pytest 对照 CPU 实现验证 `torch.allclose`                | §6.2 Step ③                               |
| ④ 准备 profiling | 生成 `profiling_file.py`，内嵌 `cases.csv`               | §6.2 Step ④                               |
| ⑤ msprof 采集    | 两次独立采集（普通 + PipeTimeline）                      | §6.2 Step ⑤                               |
| ⑥ 归档摘要       | `perf_summary.py` 生成 `summary.txt`                     | §6.2 Step ⑥                               |
| ⑦ 性能判定       | 按 8-CSV 顺序诊断，核算理论耗时                          | §7 + `references/csv_fields_reference.md` |
| ⑧ 瓶颈优化       | 查瓶颈对照表选策略                                       | `references/optimization_quickref.md`     |
| ⑨ 验证效果       | 对比 `round_NNN`，每次只改一个变量                       | §6.2 Step ⑨                               |
| ⑩ 生成报告       | HTML 报告含前后对比 + 负结果                             | §6.2 Step ⑩                               |

### 4. Step ② 具体示例：从模板生成 NPU kernel

Skill 会根据 Step ① 的模板选择，从 `references/templates/` 中取对应模板并填充参数：

```python
# Agent 自动生成（基于 template_d_rmsnorm.py）
import tilelang
import tilelang.language as T

@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def my_op_npu(D=7168, dtype="float32"):
    @T.prim_func
    def main(x: T.Buffer((4096, D), dtype),
             w: T.Buffer((D,), dtype),
             y: T.Buffer((4096, D), dtype)):
        with T.Kernel(64) as pid:  # 64 个 vec cores
            # ... 基于 template_d 的 Fragment trick + Reducer 实现
    return main
```

### 5. Step ③ 正确性验证（对照 CPU）

```python
# test_my_op.py — Agent 自动生成
import torch
from cpu_reference import my_op_cpu
from my_op_npu import my_op_npu

def test_correctness():
    for D in [4096, 7168, 8192]:
        x = torch.randn(4096, D, device="npu")
        w = torch.randn(D, device="npu")
        out_npu = my_op_npu(x, w)
        out_cpu = my_op_cpu(x.cpu(), w.cpu()).npu()
        assert torch.allclose(out_npu, out_cpu, rtol=1e-3, atol=1e-3)
```

### 6. Step ⑦⑧ 迭代优化循环

Skill 驱动的优化循环会持续到满足终止条件：

```text
Round 001 (基线): 165 us, 88% 带宽
  → ⑦ 判定: 带宽 88% > 60% → 接近硬件极限
  → ⑩ 生成报告，终止优化

或如果性能不达标：
Round 001 (基线): 300 us, 30% 带宽
  → ⑦ 诊断: vec_ratio 90% → VEC Bound
  → ⑧ 策略: T.copy DMA 替换标量读（查 optimization_quickref.md §1）
Round 002: 120 us, 55% 带宽 ✅
  → ⑦ 诊断: mte2_ratio 80% → MTE2 Bound
  → ⑧ 策略: num_stages=4 流水（查 optimization_quickref.md §2）
Round 003: 87 us, 66% 带宽 ✅
  → ⑦ 判定: 带宽 66% > 60% → 终止
```

## 关键点

- **CPU 实现的角色**：仅作为正确性参考（reference）和性能基线，不修改。
- **模板选择**：Skill 通过分析算子语义自动选择模板 A-F（§3.0 决策树）。
- **优化策略**：Skill 通过 `optimization_quickref.md` 自动映射瓶颈→策略。
- **终止条件**：带宽 > 60% 或差距 < 20% 或连续 3 次 < 5% 改善。
- **负结果记录**：所有无效尝试都被记录，避免重复踩坑。