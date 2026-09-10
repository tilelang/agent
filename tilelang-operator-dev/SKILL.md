---
name: tilelang-operator-dev
description: >
  NPU TileLang 算子自动开发与优化闭环 skill。基于 Ascend 950DT 全量 examples 归纳的方法论，
  覆盖硬件架构、编程模型、六大算子模板、瓶颈驱动优化策略、msprof 8-CSV 诊断和 10 步自动化闭环。
  当用户要求"开发 TileLang 算子"、"优化 NPU 算子"、"自动生成算子"、"算子性能调优"、
  "TileLang kernel 开发"、"NPU 算子开发"时触发。
---

# NPU TileLang 算子自动开发与优化闭环

你是 NPU TileLang 算子开发与优化专家。本 skill 提供从算子规格到高性能 NPU 实现的完整自动化闭环，
基于 Ascend 950DT 全量 examples（60+ 子目录、200+ Python 文件）归纳的方法论。

---

## 1. 硬件架构与内存层级

### 1.1 Ascend 950DT 硬件参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 芯片型号 | Ascend 950DT_9582 V120 | Da Vinci V120 架构 |
| AI Core 数 | 32 | Cube 核，用于 GEMM (T.gemm) |
| Vector Core 数 | 64 | = cube_cores × 2，`get_num_vec_cores()` |
| GM (HBM) 峰值带宽 | 4000 GB/s (4 TB/s) | HBM 3200 MHz, 96 GB |
| UB 大小 | 256 KB / core | Unified Buffer，类比 GPU shared memory |
| L1 大小 | 512 KB / core | GEMM 数据缓冲（L1 → L0A/L0B → L0C） |
| 运行频率 | 1650 MHz | — |
| SIMD 向量长度 (VL) | 64 (float32) | 每个 SIMD 寄存器 64 个 float32 lane |

### 1.2 内存层级与搬运引擎

| 层级 | TileLang 分配 | 容量 | 搬运引擎 | 说明 |
|------|-------------|------|---------|------|
| GM (HBM) | `T.Buffer` / `T.Tensor` | 96 GB | — | 全局显存，所有核可见 |
| L1 | `T.alloc_l1` | 512 KB | MTE2 (GM→L1) | GEMM 专用，Cube 单元访问 |
| UB (Shared) | `T.alloc_shared` | 256 KB | MTE2 (GM→UB), MTE3 (UB→GM) | Vector 单元访问，最常用 |
| L0A/L0B | `T.alloc_l0a/l0b` | — | MTE1 (L1→L0) | GEMM 操作数，Cube 单元私有 |
| L0C | `T.alloc_l0c` | — | FixPipe (L0C→UB/GM) | GEMM 累加器，`T.dual_copy` 搬出 |
| Fragment | `T.alloc_fragment` | 寄存器 | 直接读写 | Vector 寄存器，零延迟 |
| Reducer | `T.alloc_reducer` | — | `T.finalize_reducer` | 跨线程归约专用 |

**引擎并行性（关键优化依据）**：MTE2（读 DMA）、MTE3（写 DMA）、Cube（MAD/MTE1）、Vector（SimdVF/SimtVF）、Scalar **五类引擎可并行执行**。
`num_stages` 流水的本质就是让 MTE2 读下一帧时，Vector/Cube 计算当前帧，MTE3 写上一帧。
msprof 中各引擎 ratio 之和可 > 100%，表示流水重叠生效。

### 1.3 两种执行模型

| 执行模型 | TileLang API | 编程范式 | 适用场景 |
|---------|-------------|---------|---------|
| **SimdVF** | `T.SimdVF(latency=...)` | 纯向量 SIMD，无线程概念，用 `simd` 模块操作 SIMD 寄存器 | 规则向量计算、SIMD intrinsics（softmax、量化、topk） |
| **SimtVF** | `T.SimtVF(threads=N)` | SIMT 向量，类 GPU 线程模型，每线程处理 1 元素 | 不规则 gather、逐元素操作、需要线程级分支逻辑 |

**选择原则**：规则计算（连续向量、无分支）用 SimdVF + `simd` intrinsics 性能最优；不规则访问（gather、条件分支）用 SimtVF 更灵活。

---

## 2. 核心 API 速查

### 2.1 内存分配

| API | 说明 |
|-----|------|
| `T.alloc_shared(shape, dtype)` | UB 缓冲（Vector 用），≤ 256 KB |
| `T.alloc_l1(shape, dtype)` | L1 缓冲（Cube 用），≤ 512 KB |
| `T.alloc_l0c/l0a/l0b(shape, dtype)` | L0 级（GEMM 累加器/操作数） |
| `T.alloc_fragment(shape, dtype)` | Vector 寄存器（零延迟） |
| `T.alloc_reducer(shape, dtype, op="sum")` | 归约器 + `T.finalize_reducer` |

### 2.2 数据搬运

| API | 说明 |
|-----|------|
| `T.copy(src, dst, pad_value=...)` | DMA 搬运（自动选择 MTE2/MTE3） |
| `T.dual_copy(l0c, ub)` | L0C → UB 两步搬运（FixPipe） |
| `T.fill(buf, value)` | 批量填充 |

### 2.3 计算

| API | 说明 |
|-----|------|
| `T.gemm(a, b, c, transpose_B=True, clear_accum=...)` | Cube GEMM（L0A × L0B → L0C） |
| `T.reduce_max/sum/min(...)` | 归约操作 |

### 2.4 调度

| API | 说明 |
|-----|------|
| `T.Kernel(N) as pid` | 核间并行 |
| `T.Persistent(tiles, N, pid, group_size=, num_stages=)` | 持久化调度 + 工作窃取 |
| `T.Pipelined(n, num_stages=K)` | 软件流水（K 缓冲） |
| `T.Parallel(n)` / `T.serial(n)` | 并行 / 串行循环 |

### 2.5 同步

| API | 说明 |
|-----|------|
| `T.set_atomic("add", dtype)` | 原子累加（Split-K 用） |
| `T.ascend_sync_inter_arrive/wait(tag, flag)` | 核间屏障同步 |
| `T.pdl_sync()` / `T.sync_threads()` | 核内同步 |

### 2.6 多缓冲

| API | 说明 |
|-----|------|
| `T.annotate_buffer_versions({buf: N})` | 自动 N 版本多缓冲 |
| `T.annotate_manual_multi_buffer(buf)` | 手动 ping-pong 环（loop-carried RAW） |
| `annotations={"multi_buffer_eligible": [buf]}` | Pipelined 标记可多缓冲 |

### 2.7 动态形状

| API | 说明 |
|-----|------|
| `T.dynamic("name")` | 运行时动态维度 |
| `T.StridedTensor(shape, strides, dtype)` | 跨步张量（非连续） |
| `T.assume(cond)` / `T.assume_no_conflict(...)` | 编译器提示 |

### 2.8 SIMD Intrinsics（`tilelang.language.simd as S`）

| Intrinsic | 功能 | 典型用途 |
|-----------|------|---------|
| `S.vld / S.vsts` | 向量加载 / 存储 | UB ↔ 寄存器 |
| `S.vmax / S.vmin / S.vadd / S.vmul / S.vdiv` | 逐元素算术 | softmax、量化 |
| `S.vcmax / S.vcmin / S.vcadd` | 跨 lane 归约（reduce） | amax、sum |
| `S.vdupv` | 广播标量到所有 lane | 归约结果广播 |
| `S.vcvt(x, "float8_e4m3fn")` | 类型转换 | FP8 量化 |
| `S.vexpdif(x, max, ...)` | exp(x - max) 融合 | online softmax |
| `S.vsel / S.vcmp` | 条件选择 / 比较 | TopK 选择 |
| `S.vld2 / S.vsstb` | 交错加载 / NZ 布局存储 | Flash Attention softmax packing |
| `S.pset(32, "PAT_ALL")` | 设置 SIMD 模式 | 全 lane / 单 lane 模式 |
| `S.mem_bar("VST_VLD")` | 存储→加载屏障 | 保证写后读顺序 |

### 2.9 JIT 编译与 PassConfig

```python
@tilelang.jit(
    out_idx=[1, 2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
    compile_flags=["--cce-res-usage"],
)
def _kernel(D, dtype):
    ...
    return _prim_func
```

---

## 3. 六大算子模板

### 3.0 模板选择决策树

```
算子输入输出特征
    │
    ├─ 有 GEMM 计算？
    │   ├─ 是 → 有 softmax？
    │   │       ├─ 是 → 模板 C (Flash Attention)
    │   │       └─ 否 → 模板 B (GEMM/Cube)
    │   └─ 否 → 有归约？
    │       ├─ 是 → 有类型转换？
    │       │       ├─ 是 → 模板 E (量化/Cast)
    │       │       └─ 否 → 模板 D (归约/Norm)
    │       └─ 否 → 有 gather/scatter？
    │           ├─ 是 → 模板 A (Gather/Vector)
    │           └─ 否 → 有选择/排序？
    │               └─ 是 → 模板 F (TopK/选择)
    └─ 以上都不是 → 需要新模板
```

| 模板 | 类型 | 核心引擎 | 典型算子 | 参考文件 |
|------|------|---------|---------|---------|
| A | Gather/Vector | MTE2+SimtVF | Gather | `gather_opt.py` |
| B | GEMM/Cube | Cube+MTE2 | MatMul | `ascend/example_gemm.py` |
| C | Flash Attention | Cube+SimdVF | MHA/GQA | `ascend/flash_attention/core.py` |
| D | 归约/Norm | SimtVF+Reducer | RMSNorm | `ascend/example_rmsnorm.py` |
| E | 量化/Cast | SimdVF+simd | per_token_cast | `ascend/example_simdvf_per_token_cast_to_fp8.py` |
| F | TopK/选择 | SimdVF+simd | MoE TopK | `ascend/example_simdvf_topk_gate.py` |

### 3.1 模板 A：Gather / Vector

**核心原则：DMA 优先，避免标量 GM 访问。**

```python
@tilelang.jit(pass_configs={
    PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
})
def _kernel_ascend(D, dtype, index_dtype):
    n, m = T.dynamic("n"), T.dynamic("m")
    BD = min(tilelang.next_power_of_2(D), C)
    BM = C // BD
    num_cores = get_num_vec_cores()  # 64

    @T.prim_func
    def _kernel(X, INDEX, Y):
        with T.Kernel(num_cores) as pid:
            values_ub = T.alloc_shared([BM, BD], dtype)
            index_ub  = T.alloc_shared([BM], index_dtype)
            for block in T.Persistent([nbm * nbd], num_cores, pid,
                                       group_size=1, num_stages=4):
                T.copy(INDEX[sm:sm+copy_m], index_ub[:copy_m])
                for m_i in T.serial(BM):
                    idx = index_ub[m_i]
                    T.copy(X[idx, sd:sd+copy_d], values_ub[m_i, :copy_d])
                T.copy(values_ub[:copy_m, :copy_d], Y[sm:sm+copy_m, ...])
    return _kernel
```

**D 分支策略**：
- D ≤ 4：SimtVF 标量 gather（数据量 < 1KB，DMA 开销不划算）
- 4 < D ≤ 2048：批量 T.copy（BM 行一批），MTE2 引擎，num_stages=4 流水
- D > 2048：逐行 T.copy（BM=1），单行数据大，2D 循环

### 3.2 模板 B：GEMM / Cube

**核心：GM → L1 → L0A/L0B → L0C → UB → GM 五级流水。**

```python
@T.prim_func
def main(X: T.Buffer((M, K), dtype), W: T.Buffer((N, K), dtype), C: T.Buffer((M, N), out_dtype)):
    with T.Kernel(NUM_BLOCKS) as bx:
        res = T.alloc_l0c((TILE_M, TILE_N), "float32")
        x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
        w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
        for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
            m_tile, n_tile = aswt_swizzle(tile_idx)
            for kt in T.Pipelined(K_TILES, num_stages=2):
                T.copy(X[m_tile*TILE_M:..., kt*TILE_K:...], x_l1)
                T.copy(W[n_tile*TILE_N:..., kt*TILE_K:...], w_l1)
                T.gemm(x_l1, w_l1, res, transpose_B=True,
                       clear_accum=(kt == 0),
                       unit_flag_ctrl=T.Select(kt == K_TILES-1, UF_3, UF_2))
            T.dual_copy(res, temp)
            T.dual_copy(temp, C[...])
```

**关键技术**：
- `aswt_swizzle`：T.macro 实现的地址 swizzle，提升 L2 cache 局部性
- `unit_flag_ctrl`：控制 FixPipe 与 GEMM 的重叠时机，尾帧用 UF_3 确保写回完成
- `T.dual_copy`：L0C→UB→GM 两步搬运，用于混合精度输出
- `T.set_atomic("add")`：累加模式，C += A@B^T（Split-K 用）

### 3.3 模板 C：Flash Attention

**核心：Cube 负责 QK^T 和 PV 两次 GEMM，Vector (SimdVF) 负责 online softmax。**

```python
for kv in T.Pipelined(NUM_KV_BLOCKS, num_stages=3):
    qk(kv, K, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub)      # ① Cube: QK^T
    softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, ...)                      # ② SimdVF: online softmax + NZ packing
    pack_p(P_nz_ub, P_shared)                                               # ③ L0C→UB NZ→ND
    pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)   # ④ Cube: P@V
    accumulate_output(O_tmp_ub, O_ub, alpha_ub)                             # ⑤ SimdVF: O = alpha*O + O_tmp
```

**NZ 布局 softmax packing**：
- `T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})` 指定 NZ 布局
- `S.vcvt(e, "bfloat16", part=0/1)` 分离偶数/奇数位
- `S.vor(even, odd)` 合并为 bf16x128
- `S.vsstb(ptr, data, stride=..., update=True)` NZ 布局存储
- `S.vexpdif(x, max)` 融合 exp(x-max) 避免 overflow

### 3.4 模板 D：归约 / Norm

**核心：Fragment 寄存器复用 + Reducer 跨线程归约。**

```python
with T.SimtVF(threads=256):
    x_frag = T.alloc_fragment((TILE,), "float32")
    for i in T.Parallel(TILE):
        x_frag[i] = x_ub[i]                    # ① UB→Fragment（一次）

    sum_sq = T.alloc_reducer((1,), "float32", op="sum", replication="all")
    T.clear(sum_sq)
    for i in T.Parallel(TILE):
        sum_sq[0] += x_frag[i] * x_frag[i]     # ② 从寄存器归约（无 UB 读）
    T.finalize_reducer(sum_sq)                  # ③ 跨线程归约

    rstd = T.rsqrt(sum_sq[0] / d + eps)
    for i in T.Parallel(TILE):
        y_ub[i] = x_frag[i] * rstd * w_ub[i]    # ④ 复用 x_frag（无 x reload）
```

### 3.5 模板 E：量化 / Cast

**核心：SimdVF + simd intrinsics 实现 amax 归约 + FP8 转换。**

```python
with T.SimdVF():
    eps = S.vdup(1e-4, "float32")
    fp8_max_reg = S.vdup(448.0, "float32")
    for i in range(blk_m):
        for j in range(group_block):
            x0 = S.vld(y_ub[i, col])
            x1 = S.vld(y_ub[i, col + 64])
            amax = S.vcmax(S.vmax(S.vabs(x0), S.vabs(x1)))   # 跨 lane amax 归约
            scale = S.vdiv(S.vmax(amax, eps), fp8_max_reg)
            scale_brc = S.vdupv(scale)                        # 广播标量
            q0 = S.vdiv(x0, scale_brc)
            q0_fp8 = S.vcvt(q0, "float8_e4m3fn")              # FP8 转换
            S.vsts(y_q_ub_fp8[i, col], q0_fp8, dist="PK4_B32")
```

### 3.6 模板 F：TopK / 选择

**核心：SimdVF 选择排序循环，k 次迭代每次选出最大值。**

```python
with T.SimdVF():
    v = [S.vld(s_ub[i * VL]) for i in range(4)]
    r = [S.vci(i * VL) for i in range(4)]
    for k in range(num_topk):
        mx = S.vdupv(S.vcmax(S.vmax(S.vmax(v[0],v[1]), S.vmax(v[2],v[3]))))
        idx = S.vdupv(S.vcmin(...))
        S.vsts(out_ub[k], idx, one, "ONEPT_B32")
        for i in range(4):
            v[i] = S.vsel(neg_inf, v[i], S.vcmp(r[i], idx, "eq"))
```

### 3.7 通用设计决策清单

| 决策项 | 推荐选择 | 依据 |
|--------|---------|------|
| 核数（Vector 算子） | `get_num_vec_cores()`（= 64） | 纯 vector 算子用 vector 核 |
| 核数（GEMM 算子） | 32（cube_cores） | GEMM 受 Cube 单元限制 |
| 核间调度 | `T.Persistent` + `group_size=1` | 工作窃取，自动负载均衡 |
| UB 分配 | `T.alloc_shared`，≤ 256 KB/core | Ascend UB 上限 |
| 流水级数 | `num_stages=2~6` | GEMM 用 2，Vector/TopK 用 4~6 |
| 不规则 gather | 逐行 `T.copy` DMA | MTE2 引擎，绕过 L2 cache |
| 元数据访问 | 批量 `T.copy` 预加载到 UB | 一次 DMA 搬完小数组 |
| 边界处理 | `T.fill(values, 0)` + `if` 守卫 + `pad_value` | 非法 index 填零 |
| PassConfig | `TIR_DISABLE_VECTORIZE` + `TL_DISABLE_DATA_RACE_CHECK` | NPU 不需要向量化 lower |
| 多 dtype 支持 | `x.view(torch.int8)` 按字节处理 | 统一 kernel 适配多 dtype |
| 设备分发 | `vllm.platform` 打桩 + fallback | try import + torch.npu 检测 |

---

## 4. 高级优化技术

### 4.1 多缓冲（Double/Multi Buffering）

| 类型 | API | 原理 | 适用场景 |
|------|-----|------|---------|
| 自动多缓冲 | `T.annotate_buffer_versions({buf: N})` | 编译器自动创建 N 个版本 | 标准流水（读→算→写） |
| 手动多缓冲 | `T.annotate_manual_multi_buffer(buf)` | 用户手写 `buf[s % 2]` 索引 | loop-carried RAW + 并发 drain |

```python
# 自动多缓冲
T.annotate_buffer_versions({s_ub: 6, out_ub: 6})
for w in T.Pipelined(N, num_stages=6, annotations={"multi_buffer_eligible": [s_ub]}):
    T.copy(scores[row, :], s_ub[:num_experts])
    with T.SimdVF():
        ...
    T.copy(out_ub[:k], topk_idx[row, :])
```

### 4.2 Split-K GEMM + 核间同步

```python
with T.Kernel(NUM_BLOCKS) as bx:
    split_id = bx % split_k
    group_id = bx // split_k
    if split_id != 0:
        T.set_atomic("add", "float32")
    for out_tile in T.Serial(TILES_PER_GROUP):
        for local_kt in T.Pipelined(K_TILES_PER_SPLIT, num_stages=2):
            T.copy(X[...], x_l1); T.copy(W[...], w_l1)
            T.gemm(x_l1, w_l1, res, clear_accum=(local_kt == 0))
        with T.PerCoreTask():
            if split_id == 0:
                T.copy(res, C[...])
            T.ascend_sync_inter_arrive("PIPE_FIX", 0)
            T.ascend_sync_inter_wait("PIPE_FIX", 0)
            if split_id != 0:
                T.copy(res, C[...])
```

### 4.3 L2 Cache 旁路

对于大粒度 DMA（≥ 128B），可通过旁路 L2 cache 减少缓存污染：
- `T.copy(..., l2_cache_ctrl="NORMAL_FV")` 保留策略
- `NOTALLOC_*` 流式访问策略
- Gather 优化中，批量 T.copy 天然绕过 L2（MTE2 直达 UB）

---

## 5. 瓶颈驱动优化策略

### 5.1 优化流程

```
msprof 采集 → 识别瓶颈引擎 → 查表选策略 → 实施 → 验证闭环
```

### 5.2 瓶颈类型与优化措施对照表

| 瓶颈类型 | msprof 指标 | 优化措施 | 实际案例 |
|---------|-----------|---------|---------|
| **Vector 标量读** | `aiv_vec_ratio` > 90%, L2 hit < 50% | 替换为 T.copy DMA，增大传输粒度 | gather d=128: 296→87us (3.4x) |
| **Scalar 循环开销** | `aiv_scalar_ratio` > 90% | 批量化处理（BM 行一批），批量 T.copy 加载索引 | gather d=656: 745→449us (1.7x) |
| **MTE2 带宽极限** | `aiv_mte2_ratio` > 95%, 带宽 > 60% | 判定接近硬件极限，若 <1% 改善则停止 | gather d=14336: 72% 带宽，6 种方案均无改善 |
| **MTE2/MTE3 未重叠** | vec+scalar+mte2+mte3 ≈ 100% | 启用 num_stages=4 四缓冲流水 | 所有 DMA 分支标配 |
| **核间负载不均衡** | 各核 Task Duration 方差大 | `T.Persistent group_size=1` 工作窃取 | prepare_llama_decode |
| **元数据反复 GM 访问** | scalar_time 含大量小数组 GM 读 | 批量 T.copy 预加载到 UB | gather_*_zos 系列 |
| **嵌套串行循环** | serial 循环内含 GM 访问 | SimtVF 展平到 1D 线程 | gather_*_zos: (seq,j,r) → 1D tx |
| **GEMM Cube 未饱和** | `aic_mad_ratio` < 80% | 增大 TILE_K，aswt_swizzle 提升 L2 | TILE_K=256 (bf16) |
| **FixPipe 与 GEMM 未重叠** | `aic_fixpipe_ratio` 高 | `unit_flag_ctrl` 重叠 (UF_2/UF_3) | example_gemm.py |

### 5.3 带宽利用率基准

| 带宽利用率 | 评价 | 行动 |
|-----------|------|------|
| > 60% | 接近硬件极限 | 停止优化，记录已达极限 |
| 30-60% | 有优化空间 | 检查是否 Vector/Scalar 瓶颈阻塞了 DMA |
| < 30% | 严重未利用 | 必然存在非 DMA 瓶颈，优先消除 |

### 5.4 不同算子类型的预期 ratio 分布

| 算子类型 | 主导流水 | 预期 ratio | 异常信号 |
|---------|---------|-----------|---------|
| Elementwise | VEC | vec_ratio 50-80% | MTE2 > VEC |
| Reduction | VEC | vec_ratio 40-70% | scalar > 20% |
| Activation | VEC | vec_ratio 60-85% | 大量 cast |
| MatMul | CUBE | cube_ratio 40-70% | vec > cube |
| 纯搬运 | MTE2/MTE3 | mte2+mte3 > 50% | VEC > 30% |

---

## 6. 10 步自动化闭环

### 6.1 闭环架构

```
① 分析算子规格 → ② 编写初始版本 → ③ 正确性验证 → ④ 准备 profiling → ⑤ msprof 采集
                                                                        ↓
⑩ 生成报告 ← ⑨ 验证优化效果 ← ⑧ 瓶颈定位优化 ← ⑦ 性能标准判定 ← ⑥ 归档+摘要
```

### 6.2 十步详解

#### Step ①：分析算子规格

| 输入 | 内容 | Agent 动作 |
|------|------|-----------|
| 算子语义 | Python 参考实现 / PyTorch 等价操作 / 数学公式 | 理解计算逻辑 |
| 输入输出 | shape、dtype、device | 确定 buffer 类型 |
| 关键维度 | D 的典型取值范围 | 规划分支策略 |
| 访问模式 | 连续 / 不规则 gather / GEMM / 归约 / 量化 | 选择模板 A~F |
| GPU 参考实现 | 已有的 GPU kernel | 正确性对照 + 性能基线 |

**输出**：算子类型标签（A~F）、预期主导流水（VEC/CUBE/MTE2）、分支策略。

#### Step ②：编写初始版本

按 §3 对应模板生成代码，自动完成：
- 设置 `pass_configs`（`TIR_DISABLE_VECTORIZE` + `TL_DISABLE_DATA_RACE_CHECK` + `TL_ENABLE_FAST_MATH`）
- 计算 tiling 参数（TILE_M/N/K / BD / BM / num_cores）
- 根据算子类型选择执行模型（SimdVF / SimtVF / Cube）
- 添加边界保护（copy_m / copy_d / T.fill / pad_value）
- 编写入口函数（view int8 + 设备分发 + vllm.platform 打桩）
- 编写 pytest 正确性测试和 benchmark
- 编写 profiling 入口脚本（内嵌 cases.csv，--case-id 逐场景运行）

#### Step ③：正确性验证

```bash
python ssh_remote.py sync
python ssh_remote.py exec "cd <remote_dir> && python -m pytest <test_file> -k test_<kernel> -x -v"
```

**判定**：所有 parametrize case PASSED。若失败，Agent 分析错误信息，修正 kernel 逻辑后重试。

#### Step ④：准备并校验 profiling 入口

从当前源码**重新生成**独立 `profiling_file`（禁止复用旧文件测量新 kernel）。该文件必须：
- 内嵌最终 `cases.csv`（每个 case 一行：shape、dtype、case_id）
- 通过 `--case-id` 一次只运行一个 case
- 显式调用文件内的本地目标 kernel

```bash
python <profiling_file> --case-id <case_id>
```

**校验要点**：确认输出的 kernel name，后续 msprof 的 `--kernel-name` 必须与此精确匹配。

#### Step ⑤：msprof op 采集

对每个 case 启动**两次独立采集**：

```bash
# ① 普通采集
msprof op --warm-up=10 --launch-count=5 --output=<case_output> \
  --kernel-name=<expected_kernel> \
  python <profiling_file> --case-id <case_id>

# ② PipeTimeline 采集（独立目录，不用于计时）
msprof op --warm-up=10 --launch-count=1 --replay-mode=kernel \
  --aic-metrics=PipeTimeline --output=<case_output>_pipe_timeline \
  --kernel-name=<expected_kernel> \
  python <profiling_file> --case-id <case_id>
```

| 参数 | 说明 | 何时使用 |
|------|------|---------|
| `--warm-up=10` | 预热 10 次后采集 | **始终使用**，避免 DVFS 影响 |
| `--launch-count=5` | 运行 5 次取均值 | 需要统计稳定性时 |
| `--kernel-name` | 只采集目标 kernel | **必须使用**，防止辅助 kernel 混入 |
| `--aic-metrics=PipeTimeline` | 流水时序和气泡 | 独立目录、launch-count=1，不用于计时 |

#### Step ⑥：归档数据 + 生成统计摘要

```bash
OPPROF_DIR=$(ls -td <output_dir>/OPPROF_* | head -1)
python3 {skill_path}/scripts/perf_summary.py $OPPROF_DIR <variant_output_dir> \
  --kernel-name <expected_kernel> --round-name <case_or_round_name>
```

归档目录结构：
```
<variant_output_dir>/docs/perf/
├── round_001/                    # 基线
│   ├── OpBasicInfo.csv           # 8 个原始 CSV
│   ├── PipeUtilization.csv
│   ├── ArithmeticUtilization.csv
│   ├── Memory.csv
│   ├── MemoryL0.csv
│   ├── MemoryUB.csv
│   ├── L2Cache.csv
│   ├── ResourceConflictRatio.csv
│   ├── summary.txt               # 统计摘要
│   └── pipe_timeline/            # PipeTimeline 时序数据
├── round_002/                    # 优化后
│   └── ...
```

#### Step ⑦：性能标准判定

Agent 分析流程（7 步）：
1. 读 `summary.txt` — 获取全局概览
2. 结合 `csv_fields_reference.md` — 理解各指标含义和阈值
3. 发现异常时读原始 CSV — 如核间不均衡，查看逐核数据
4. 读 `pipe_timeline/` — 分析流水先后、重叠、等待和气泡
5. 核算搬运量与计算量 — 从源码枚举 GM 读写张量、Cube/Vector 计算量
6. 结合 `optimization_quickref.md` — 将瓶颈映射为 TileLang 修改方向
7. 用中文输出分析文件

**理论耗时计算**：
```python
# 搬运理论耗时
理论耗时(us) = 实际搬运数据量(Byte) / GM 峰值带宽
# Ascend 950DT: GM 峰值带宽 ≈ 4 TB/s = 4000 GB/s

# 计算理论耗时
理论耗时(us) = 实际计算量(FLOP) / 对应单元理论算力
# cube_fp16_bf16: 432 TFLOPS  cube_fp32: 26 TFLOPS  cube_int8_fp8: 864 TOPS
# vector_fp32_add: 13.5 TFLOPS  vector_fp16_bf16_fma: 54 TFLOPS
```

**性能判定**：
- 差距 < 20% → 性能达标（接近硬件极限）
- 差距 20-50% → 有优化空间
- 差距 > 50% → 严重瓶颈，必须优化

**各指标达标标准**：

| 指标 | 达标 | 警告 | 严重问题 |
|------|------|------|---------|
| 核间负载均衡 | 各核差异 < 10% | 10-30% | > 30% |
| Block Dim | 等于可用核数 | 远小于核数 | = 1 |
| VEC ratio | 与算子类型匹配 | > 80% | > 90% 且无优化空间 |
| MTE2 ratio | < 30%（计算型） | 30-50% | > 50% |
| fixpipe ratio | < 5% | 5-15% | > 15% |
| icache miss rate | < 5% | 5-15% | > 15% |
| bank conflict | < 5% | 5-15% | > 15% |
| L2 Cache 命中率 | > 80% | 50-80% | < 50% |
| 带宽利用率 | > 60% | 30-60% | < 30% |

**PipeTimeline 流水气泡分析**：
- 气泡率 = 流水空闲时间 / 关键路径时间
- 重叠率 = 两流水重叠时间 / 较短流水活跃时间
- **流水重叠快捷判据**：`vec + scalar + mte2 + mte3` 四个 ratio 加起来 ≈ 100% 说明完全串行、该上双缓冲；加到 130% 以上才算真重叠。

#### Step ⑧：瓶颈定位与优化

确认瓶颈类型后，查阅 `references/optimization_quickref.md`，结合目标 TileLang 源码制定具体优化方法。

**瓶颈快速查找表**：

| 瓶颈类型 | 判定条件 | 首选优化 |
|---------|---------|---------|
| VEC Bound | `aiv_vec_ratio` 最高 | ① UB 融合 ② 寄存器复用 ③ 减少 Cast ④ 优化归约 ⑤ 调整 SimtVF threads |
| MTE2/MTE3 Bound | `ai*_mte2_ratio` 最高 | ① 减少 GM 往返 ② 增大 Tile ③ pad_value ④ 重用常量 ⑤ num_stages ⑥ l2_cache_ctrl |
| CUBE Bound | `aic_cube_ratio` 最高 | ① 调整 TILE_M/N/K ② L1 复用 ③ L0 累加 ④ K 流水 ⑤ 持久化 ⑥ 输出路径 |
| SCALAR Bound | `ai*_scalar_ratio` > 30% | ① 编译期特化 ② 移出循环不变量 ③ 减少 T.serial ④ 合并小 Tile |
| 核间不均衡 | 各核耗时差异 > 10% | ① 检查 T.Kernel 匹配 ② T.Persistent 工作窃取 |
| Bank Conflict | `aiv_vec_total_cflt_ratio` > 5% | ① 调整 UB shape/padding ② 改变 stride ③ 添加 padding |
| 流水重叠不足 | ratio 和 ≈ 100% | ① `T.annotate_buffer_versions` ② `T.annotate_manual_multi_buffer` |
| L2 Cache 命中率低 | `ai*_total_hit_rate` < 50% | ① Tile + 持久化 ② `l2_cache_ctrl` 策略 ③ A/B 测试 |

**交叉关联诊断**：

| 现象组合 | 根因假设 | 优先检查 |
|---------|---------|---------|
| 高 vec_ratio + 高 bank conflict | UB 布局放大 Vector 耗时 | `T.alloc_shared` shape/padding |
| 高 mte2_time + 低 L2 hit rate | 数据复用或 L2 策略不合理 | Tile 调度、`l2_cache_ctrl` |
| 高 fixpipe_ratio | 输出路径低效 | 输出 Tile、`T.dual_copy` 路径 |
| 高 mte2 + 高 mte3 | GM 双向搬运饱和 | 融合中间结果、增大 Tile |
| 低 Block Dim + 高 Duration | 并行不足 | `T.Kernel` block 数、Tile shape |
| scalar 高 + 小 shape | 动态控制占比高 | 编译期特化、减少 `T.serial` |

**仓库实现参考**：

| 优化模式 | 参考文件 |
|---------|---------|
| Vector Tile、UB 搬运、流水 | `vllm-ascend/softmax.py` |
| 动态 shape、UB/L1 流水 | `quant/cast_back_asc.py` |
| Buffer 多版本 | `vllm-ascend/cache.py` |
| Fragment 复用和 Reduction | `ascend/example_rmsnorm.py` |
| L1/L0C/GEMM | `ascend/example_gemm.py` |
| L2 旁路 | `ascend/example_gemm_bypass_l2.py` |
| 自动与手动多缓冲 | `ascend/example_manual_multibuffer.py` |
| Cube+Vector + NZ 布局 | `ascend/flash_attention/core.py` |
| SimdVF 选择排序 | `ascend/example_simdvf_topk_gate.py` |

**抽取优化模式时必须保留目标算子的接口、数据布局、边界处理和精度语义，不能整文件替换目标 kernel。**

#### Step ⑨：验证优化效果

每次优化后，重新运行 Step ⑤⑥。对比两轮摘要：

```bash
diff <variant_output_dir>/docs/perf/round_001/summary.txt \
     <variant_output_dir>/docs/perf/round_002/summary.txt
```

**对比要点**：
1. Task Duration 是否下降
2. 瓶颈单元的 ratio 是否改善
3. 核间均衡是否改善
4. 是否引入新的瓶颈

**每次只改变一个主要变量**，并使用相同 `cases.csv`、相同 kernel 过滤规则和相同计时口径。

**迭代终止条件**：
- 带宽利用率 > 60% **或**
- 实际耗时 vs 理论耗时差距 < 20% **或**
- 连续 3 次优化 < 5% 改善 **或**
- 编译失败（回退该参数）

#### Step ⑩：生成优化报告

自动生成 HTML 报告，包含：
- 各场景优化前后对比（耗时、加速比、带宽利用率、各引擎 ratio 变化）
- 与 GPU 性能对比（差距倍数）
- 使用的优化手段及效果（含瓶颈对照表映射）
- 瓶颈分析过程（msprof 8-CSV 指标变化、PipeTimeline 气泡分析）
- 理论耗时 vs 实际耗时对比
- 负结果记录（尝试过但无效的方案）
- 归档数据路径

---

## 7. msprof 8-CSV 联合诊断顺序

Agent 按**固定顺序**读取 8 个 CSV，逐步缩小瓶颈范围：

| 顺序 | CSV 文件 | 分析目标 | 关键字段 |
|------|---------|---------|---------|
| 1 | `OpBasicInfo.csv` | 校验 kernel 名/频率/Task Duration | Task Duration, Block Dim |
| 2 | `PipeUtilization.csv` | 主导流水/核间差异 | aiv_vec_ratio, aic_cube_ratio, ai*_mte2_ratio |
| 3 | `ArithmeticUtilization.csv` | 有效计算/指令结构 | cube_fops, vec_fp32_ratio |
| 4 | `Memory.csv` | 总搬运量/带宽利用率 | GM_to_UB, bandwidth_util |
| 5 | `MemoryL0.csv` | CUBE L0A/L0B/L0C 带宽 | L0A read, L0C write |
| 6 | `MemoryUB.csv` | Vector UB 读写带宽 | UB read/write bw |
| 7 | `L2Cache.csv` | 缓存命中假设验证 | total_hit_rate |
| 8 | `ResourceConflictRatio.csv` | Bank conflict/等待比例 | vec_cflt_ratio, mte2_wait_ratio |

**第 9 步**：将结论映射到 `T.Kernel`、Tile shape、`T.copy`、buffer scope、`T.Pipelined`、`T.Persistent`、`T.Parallel` 或 `T.gemm` 的具体修改。

---

## 8. 核心经验总结

### 8.1 五条黄金法则

1. **DMA 优先**：所有 GM 访问用 `T.copy`（MTE2/MTE3 引擎），**绝不**用 SimtVF 标量读 GM。唯一例外：数据量 < 1KB。
2. **批量摊薄**：索引加载、输出写入都要批量化（BM 行一批），减少 scalar 循环开销和 DMA 次数。
3. **流水重叠**：`num_stages=2~6` 多缓冲，让 MTE2 读和 MTE3 写并行重叠。检查各引擎 ratio 之和 > 100% 确认流水生效。
4. **引擎匹配**：GEMM 用 Cube（L1/L0），向量计算用 Vector（UB/Fragment），两者通过 `T.dual_copy` 衔接。
5. **寄存器复用**：用 `T.alloc_fragment` 缓存中间结果，避免重复 UB 加载。归约用 `T.alloc_reducer + T.finalize_reducer`。

### 8.2 四个陷阱

1. **盲目调参**：不先 msprof 定位瓶颈就调参数，大概率无效。先定位瓶颈引擎再针对性优化。
2. **忽视 Scalar 开销**：NPU 的 Scalar 循环控制 + 索引加载开销可达 99.7%，必须通过批量化摊薄。
3. **过度优化已达极限**：带宽利用率 > 60% 时，多种方案可能都 < 1% 改善。学会判定硬件极限。
4. **忽视 NZ 布局**：Flash Attention 中 softmax 输出必须以 NZ 布局存储，否则第二次 GEMM 需额外转换。

### 8.3 注意事项

1. **必须 warm-up**：首次运行受 DVFS 影响，耗时偏高。始终使用 `--warm-up=10`
2. **频率检查**：读取 `OpBasicInfo.csv` 的 `Current Freq` 和 `Rated Freq`，若 Current < Rated 说明芯片未满频
3. **MTE2/MTE3 带宽共享**：同时读写 GM 时，总带宽被共享
4. **PipeTimeline 不计时**：动态插桩会改变耗时，只用于流水分析
5. **小数据量场景**：数据量很小时头开销占比会很高，这不一定是算子问题
6. **每次只改一个变量**：修改 Tile/buffer/stage 后先跑精度回归，再重新采集
7. **UB/L1/L0 容量检查**：Tile 大小、buffer 数量和 pipeline stage 必须满足目标设备容量

---

## 9. Agent Prompt 模板

```markdown
你是 NPU TileLang 算子优化专家。请按以下 10 步工作：

## 输入
- 算子名：{name}
- 参考实现：{reference_impl}
- 关键维度：{dimensions}（如 d=4/128/656/14336）
- GPU kernel：{gpu_kernel_path}（用于正确性对照和性能基线）
- 远程 NPU 环境：{ssh_config}
- ops-profiling skill 路径：{skill_path}

## 步骤
① 分析算子类型，选择模板（A: Gather / B: GEMM / C: FlashAttn / D: Norm / E: Quant / F: TopK）
② 按 §3 对应模板编写初始 NPU kernel + 测试 + profiling 脚本（内嵌 cases.csv）
③ ssh_remote.py sync → run 正确性测试 → 修复直至全部 PASSED
④ 准备 profiling 入口，触发 JIT 确认目标 kernel 运行
⑤ 对每个 case 跑 msprof op（普通 + PipeTimeline 两次采集）
⑥ perf_summary.py 归档 CSV + 生成 summary.txt
⑦ 读 summary.txt → 按 8-CSV 联合诊断顺序分析 → 核算搬运/计算量 → 判定达标
⑧ 按 §5.2 瓶颈对照表选择优化策略，结合源码制定修改方案
⑨ 修改代码 → 回到 ③ 正确性验证 → ⑤ 重新采集 → ⑥ 归档 → 对比 round_NNN
⑩ 迭代至带宽 > 60% 或差距 < 20% 或连续 3 次 < 5% 改善
   生成 HTML 优化报告

## 约束
- GPU kernel 不修改
- 每次修改后必须验证正确性（pytest Level 1 精度回归）
- 每次只改变一个主要变量
- 记录所有尝试过的方案（含负结果）
- vllm.platform 打桩 + fallback
- 代码长度控制在与 GPU kernel 2x 以内
- 分析交付必须使用中文
- --kernel-name 必须与 profiling_file 中的目标 kernel 精确匹配
- 禁止复用旧 profiling_file 测量新 kernel
```

---

## 10. 所需工具链

| 工具 | 用途 | 关键参数/路径 |
|------|------|-------------|
| `ssh_remote.py` | 本地→远程同步、测试、执行 | sync / run / bench / exec |
| `msprof op` | NPU 性能采集 | `--kernel-name`（必须）、`--warm-up=10`、`--launch-count=5` |
| `perf_summary.py` | 归档 CSV + 生成摘要 | `--kernel-name`、`--round-name` |
| `profiling_file.py` | profiling 入口脚本 | `--case-id` 逐场景，内嵌 cases.csv |
| `tile_kernels.config` | 核数查询 | `get_num_vec_cores()` |
| `pytest` | 正确性 + benchmark | `--run-benchmark` |
| `csv_fields_reference.md` | 8 个 CSV 字段定义和阈值 | Step ⑦ 分析时查阅 |
| `optimization_quickref.md` | 瓶颈→TileLang 优化映射 | Step ⑧ 定位瓶颈后查阅 |

---

## 11. 仓库实现参考索引

### 11.1 Ascend NPU 专用示例（`examples/ascend/`）

| 类别 | 文件 | 关键技术 |
|------|------|---------|
| GEMM | `example_gemm.py` | auto-schedule, aswt_swizzle, unit_flag_ctrl, dual_copy, hf32 |
| GEMM | `example_gemm_l0.py` | 显式 L0A/L0B/L0C, L1→L0 子循环 |
| GEMM | `example_gemm_bypass_l2.py` | L2 旁路 |
| GEMM | `example_gemm_splitk.py` | Split-K, inter-core sync, atomic, deterministic |
| GEMM | `example_gemm_ub_merge.py` | UB merge |
| GEMM | `example_blockscaled_gemm.py` | MXFP8/MXFP4 block-scaled |
| Flash Attn | `flash_attention/core.py` | Cube+Vector, NZ layout, SIMD softmax packing |
| Flash Attn | `flash_attention/example_mha.py / gqa.py` | MHA/GQA 前向 |
| Norm | `example_rmsnorm.py` | Fragment trick, reducer, double buffer, 64 cores |
| Atomic | `example_atomic.py` | GM atomic add/max/min |
| Compress | `example_compress.py` | 两阶段 compress + state-cache, dynamic, StridedTensor |
| 量化 | `example_simdvf_per_token_cast_to_fp8.py` | SimdVF, amax 归约, vcvt FP8 |
| TopK | `example_simdvf_topk_gate.py` | MoE topk gate, 选择排序, pad_value |
| VecAdd | `example_simdvf_vecadd.py` | SimdVF 纯向量 |
| VecAdd | `example_simtvf_vecadd.py` | SimtVF SIMT 向量 |
| 调度 | `example_while_pipelined.py` | while 循环自动流水 |
| 调度 | `example_manual_multibuffer.py` | 手动 ping-pong 多缓冲 |

### 11.2 TileKernels-Nightly 算子库

| 模块 | 代表文件 | 说明 |
|------|---------|------|
| vllm-ascend | `gather_opt.py` | Gather 优化版 |
| vllm-ascend | `rope.py` | RoPE 旋转位置编码 |
| vllm-ascend | `softmax.py` | Softmax / LogSoftmax |
| vllm-ascend | `norm.py` | RMSNorm/LayerNorm |
| vllm-ascend | `cache.py` | KV cache |
| quant | `per_token_cast_*_asc.py` | per-token FP8 cast |
| quant | `per_block_cast_*_asc.py` | per-block cast |
| moe | `moe_topk_gate_*_asc.py` | MoE topk gate |
