# NPU TileLang 算子开发与优化方法论

基于 Ascend 950DT 全量 examples 目录（60+ 子目录、200+ Python 文件）的归纳总结  
涵盖 GEMM、Flash Attention、RMSNorm、量化、TopK、Gather、Compress 等全算子类型

---

## 一、NPU 硬件架构与内存层级

### 1.1 Ascend 950DT 硬件参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 芯片型号 | Ascend 950DT_9582 V120 | Da Vinci V120 架构 |
| AI Core 数 | 32 | Cube 核，用于 GEMM (T.gemm) |
| Vector Core 数 | 64 | = cube_cores × 2，纯向量算子用 `get_num_vec_cores()` |
| GM (HBM) 峰值带宽 | 4000 GB/s (4 TB/s) | HBM 3200 MHz, 96 GB |
| UB 大小 | 256 KB / core | Unified Buffer，类比 GPU shared memory |
| L1 大小 | 512 KB / core | GEMM 数据缓冲（L1 → L0A/L0B → L0C） |
| 运行频率 | 1650 MHz | — |
| SIMD 向量长度 (VL) | 64 (float32) | 每个 SIMD 寄存器 64 个 float32 lane |

### 1.2 内存层级与数据搬运引擎

NPU 有**四层内存**和**三类搬运引擎**，理解引擎与层级的映射关系是优化的基础：

| 层级 | TileLang 分配 | 容量 | 搬运引擎 | 说明 |
|------|---------------|------|----------|------|
| GM (HBM) | `T.Buffer` / `T.Tensor` | 96 GB | — | 全局显存，所有核可见 |
| L1 | `T.alloc_l1` | 512 KB | 🟧 MTE2 (GM→L1) | GEMM 专用，Cube 单元访问 |
| UB (Shared) | `T.alloc_shared` | 256 KB | 🟧 MTE2 (GM→UB), 🟧 MTE3 (UB→GM) | Vector 单元访问，最常用 |
| L0A/L0B | `T.alloc_l0a/l0b` | — | MTE1 (L1→L0) | GEMM 操作数，Cube 单元私有 |
| L0C | `T.alloc_l0c` | — | FixPipe (L0C→UB/GM) | GEMM 累加器，`T.dual_copy` 搬出 |
| Fragment | `T.alloc_fragment` | 寄存器 | 直接读写 | Vector 寄存器，零延迟 |
| Reducer | `T.alloc_reducer` | — | `T.finalize_reducer` | 跨线程归约专用 |

> **引擎并行性（关键优化依据）**  
> MTE2（读 DMA）、MTE3（写 DMA）、Cube（MAD/MTE1）、Vector（SimdVF/SimtVF）、Scalar **五类引擎可并行执行**。  
> `num_stages` 流水的本质就是让 MTE2 读下一帧时，Vector/Cube 计算当前帧，MTE3 写上一帧。  
> msprof 中各引擎 ratio 之和可 > 100%，表示流水重叠生效。

### 1.3 两种执行模型

| 执行模型 | TileLang API | 编程范式 | 适用场景 | 示例 |
|----------|--------------|----------|----------|------|
| 🟩 SimdVF | `T.SimdVF(latency=...)` | 纯向量 SIMD，无线程概念<br>用 `simd` 模块直接操作 SIMD 寄存器 | 规则向量计算、SIMD intrinsics<br>（softmax、量化、topk） | `example_simdvf_vecadd.py`<br>`flash_attention/core.py`<br>`example_simdvf_topk_gate.py` |
| 🟩 SimtVF | `T.SimtVF(threads=N)` | SIMT 向量，类 GPU 线程模型<br>每线程处理 1 元素，`T.Parallel` 并行 | 不规则 gather、逐元素操作<br>需要线程级分支逻辑 | `example_simtvf_vecadd.py`<br>`gather_opt.py`<br>`example_rmsnorm.py` |

**选择原则：** 规则计算（连续向量、无分支）用 SimdVF + `simd` intrinsics 性能最优；不规则访问（gather、条件分支）用 SimtVF 更灵活。

---

## 二、TileLang NPU 编程模型

### 2.1 核心 API 速查

| 类别 | API | 说明 |
|------|-----|------|
| **内存分配** | `T.alloc_shared(shape, dtype)` | UB 缓冲（Vector 用），≤ 256 KB |
| | `T.alloc_l1(shape, dtype)` | L1 缓冲（Cube 用），≤ 512 KB |
| | `T.alloc_l0c/l0a/l0b(shape, dtype)` | L0 级（GEMM 累加器/操作数） |
| **寄存器** | `T.alloc_fragment(shape, dtype)` | Vector 寄存器（零延迟） |
| | `T.alloc_reducer(shape, dtype, op="sum")` | 归约器 + `T.finalize_reducer` |
| **数据搬运** | `T.copy(src, dst, pad_value=...)` | DMA 搬运（自动选择 MTE2/MTE3） |
| | `T.dual_copy(l0c, ub)` | L0C → UB 两步搬运（FixPipe） |
| | `T.fill(buf, value)` | 批量填充 |
| **计算** | `T.gemm(a, b, c, transpose_B=True, clear_accum=...)` | Cube GEMM（L0A × L0B → L0C） |
| | `T.reduce_max/sum/min(...)` | 归约操作 |
| **调度** | `T.Kernel(N) as pid` | 核间并行 |
| | `T.Persistent(tiles, N, pid, group_size=, num_stages=)` | 持久化调度 + 工作窃取 |
| | `T.Pipelined(n, num_stages=K)` | 软件流水（K 缓冲） |
| | `T.Parallel(n)` / `T.serial(n)` | 并行 / 串行循环 |
| **同步** | `T.set_atomic("add", dtype)` | 原子累加（Split-K 用） |
| | `T.ascend_sync_inter_arrive/wait(tag, flag)` | 核间屏障同步 |
| | `T.pdl_sync()` / `T.sync_threads()` | 核内同步 |
| **多缓冲** | `T.annotate_buffer_versions({buf: N})` | 自动 N 版本多缓冲 |
| | `T.annotate_manual_multi_buffer(buf)` | 手动 ping-pong 环（loop-carried RAW） |
| | `annotations={"multi_buffer_eligible": [buf]}` | Pipelined 标记可多缓冲 |
| **动态形状** | `T.dynamic("name")` | 运行时动态维度 |
| | `T.StridedTensor(shape, strides, dtype)` | 跨步张量（非连续） |
| | `T.assume(cond)` / `T.assume_no_conflict(...)` | 编译器提示 |

### 2.2 SIMD Intrinsics 速查（`tilelang.language.simd as S`）

| Intrinsic | 功能 | 典型用途 |
|-----------|------|----------|
| `S.vld` / `S.vsts` | 向量加载 / 存储 | UB ↔ 寄存器 |
| `S.vmax` / `S.vmin` / `S.vadd` / `S.vmul` / `S.vdiv` | 逐元素算术 | softmax、量化 |
| `S.vcmax` / `S.vcmin` / `S.vcadd` | 跨 lane 归约（reduce） | amax、sum |
| `S.vdupv` | 广播标量到所有 lane | 归约结果广播 |
| `S.vcvt(x, "float8_e4m3fn")` | 类型转换 | FP8 量化 |
| `S.vexpdif(x, max, ...)` | exp(x - max) 融合 | online softmax |
| `S.vsel` / `S.vcmp` | 条件选择 / 比较 | TopK 选择 |
| `S.vld2` / `S.vsstb` | 交错加载 / NZ 布局存储 | Flash Attention softmax packing |
| `S.pset(32, "PAT_ALL")` | 设置 SIMD 模式 | 全 lane / 单 lane 模式 |
| `S.mem_bar("VST_VLD")` | 存储→加载屏障 | 保证写后读顺序 |

### 2.3 JIT 编译与 PassConfig

```python
@tilelang.jit(
    out_idx=[1, 2],                          # 输出 buffer 索引
    pass_configs={
        PassConfigKey.TL_ENABLE_FAST_MATH: True,           # 快速数学
        PassConfigKey.TIR_DISABLE_VECTORIZE: True,         # NPU 不需要向量化 lower
        PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,    # NPU 不需要数据竞争检查
    },
    compile_flags=["--cce-res-usage"],        # CCE 编译器选项
)
def _kernel(D, dtype):
    ...
    return _prim_func
```

---

## 三、如何从零编写高性能 NPU TileLang 算子

### 3.1 算子分类与模板选择

根据全量 examples 分析，NPU 算子可分为**六大类**，每类有对应的编写模板：

| 类型 | 代表算子 | 核心引擎 | 模板 | 示例文件 |
|------|----------|----------|------|----------|
| **A. Gather/Vector** | gather、gather_input_ids_zos | 🟧 MTE2 + 🟩 SimtVF | §3.2 | `gather_opt.py` |
| **B. GEMM/Cube** | matmul、split-K GEMM | 🔵 Cube + 🟧 MTE2 | §3.3 | `ascend/example_gemm.py` |
| **C. Flash Attention** | MHA、GQA 前向 | 🔵 Cube + 🟩 SimdVF | §3.4 | `ascend/flash_attention/core.py` |
| **D. 归约/Norm** | RMSNorm、LayerNorm | 🟩 SimtVF + Reducer | §3.5 | `ascend/example_rmsnorm.py` |
| **E. 量化/Cast** | per-token FP8 cast | 🟩 SimdVF + `simd` | §3.6 | `ascend/example_simdvf_per_token_cast_to_fp8.py` |
| **F. TopK/选择** | MoE topk gate | 🟩 SimdVF + `simd` | §3.7 | `ascend/example_simdvf_topk_gate.py` |

### 3.2 模板 A：Gather / Vector 类算子

**核心原则：DMA 优先，避免标量 GM 访问。** NPU 没有 TMA 硬件 gather，必须用 `T.copy` DMA（MTE2 引擎）替代 SimtVF 标量读。

| 访问模式 | GPU 方案 | NPU 方案 | NPU 性能 |
|----------|----------|----------|----------|
| 连续 DMA 搬运 | TMA / cp.async | ✅ T.copy（MTE2/MTE3） | 接近 GPU |
| 不规则行 gather | TMA 硬件 gather | ✅ 逐行 T.copy（每行连续） | 2~4x 慢 |
| 逐元素标量 gather | thread 读 GM | ❌ SimtVF 标量读（L2 miss） | 10x+ 慢 |

```python
@tilelang.jit(pass_configs={PassConfigKey.TIR_DISABLE_VECTORIZE: True, ...})
def _kernel_ascend(D, dtype, index_dtype):
    n, m = T.dynamic("n"), T.dynamic("m")
    BD = min(tilelang.next_power_of_2(D), C)      # ① 自适应 tiling
    BM = C // BD
    num_cores = get_num_vec_cores()                 # ② 纯 vector 算子用 vec_cores

    @T.prim_func
    def _kernel(X, INDEX, Y):
        with T.Kernel(num_cores) as pid:
            values_ub = T.alloc_shared([BM, BD], dtype)
            index_ub  = T.alloc_shared([BM], index_dtype)
            for block in T.Persistent([nbm * nbd], num_cores, pid,
                                       group_size=1, num_stages=4):  # ③ 持久化+流水
                T.copy(INDEX[sm:sm+copy_m], index_ub[:copy_m])      # ④ 批量加载索引
                for m_i in T.serial(BM):
                    idx = index_ub[m_i]
                    T.copy(X[idx, sd:sd+copy_d], values_ub[m_i, :copy_d])  # ⑤ 逐行 DMA gather
                T.copy(values_ub[:copy_m, :copy_d], Y[sm:sm+copy_m, ...])  # ⑥ 批量写回
    return _kernel
```

**D 分支策略：**

| D 范围 | 策略 | 理由 |
|--------|------|------|
| D ≤ 4 | SimtVF 标量 gather | 数据量 < 1KB，DMA 开销不划算 |
| 4 < D ≤ 2048 | ✅ 批量 T.copy（BM 行一批） | MTE2 引擎，num_stages=4 流水 |
| D > 2048 | 逐行 T.copy（BM=1） | 单行数据大，2D 循环 |

### 3.3 模板 B：GEMM / Cube 类算子

**核心：GM → L1 → L0A/L0B → L0C → UB → GM 五级流水。**  
`T.gemm` 自动处理 L0→L0C 的 Cube 计算，用户负责 GM→L1 的 MTE2 搬运和 L0C→GM 的 FixPipe 写回。

```python
@T.prim_func
def main(X: T.Buffer((M, K), dtype), W: T.Buffer((N, K), dtype), C: T.Buffer((M, N), out_dtype)):
    with T.Kernel(NUM_BLOCKS) as bx:
        res = T.alloc_l0c((TILE_M, TILE_N), "float32")     # L0C 累加器
        x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)          # L1 缓冲
        w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
        for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
            m_tile, n_tile = aswt_swizzle(tile_idx)          # ① swizzle 提升 L2 局部性
            for kt in T.Pipelined(K_TILES, num_stages=2):    # ② K 维流水
                T.copy(X[m_tile*TILE_M:..., kt*TILE_K:...], x_l1)  # ③ GM→L1 (MTE2)
                T.copy(W[n_tile*TILE_N:..., kt*TILE_K:...], w_l1)
                T.gemm(x_l1, w_l1, res, transpose_B=True,            # ④ Cube GEMM
                       clear_accum=(kt == 0),
                       unit_flag_ctrl=T.Select(kt == K_TILES-1, UF_3, UF_2))  # ⑤ 重叠 FixPipe
            if MIXED:
                T.dual_copy(res, temp)                        # ⑥ L0C→UB→GM (FixPipe)
                T.dual_copy(temp, C[...])
            else:
                T.copy(res, C[...])
```

**关键技术：**
- `aswt_swizzle`：T.macro 实现的地址 swizzle，提升 L2 cache 局部性（见 `example_gemm.py:40`）
- `unit_flag_ctrl`：控制 FixPipe 与 GEMM 的重叠时机，尾帧用 UF_3 确保写回完成
- `T.dual_copy`：L0C→UB→GM 两步搬运，用于混合精度输出（fp32 累加→bf16 输出）
- `T.set_hf32_mode`：FP32 GEMM 的 HF32 模式（`"nearest_even"` / `"nearest_zero"`）
- `T.set_atomic("add")`：累加模式，C += A@B^T（Split-K 用）

### 3.4 模板 C：Flash Attention（Cube + Vector 协作）

**核心：Cube 负责 QK^T 和 PV 两次 GEMM，Vector (SimdVF) 负责 online softmax。**  
难点在于 softmax 输出需要以 **NZ 布局**直接写入 UB，供第二次 GEMM 消费，避免 ND→NZ 转换。

```python
# 结构：每个 KV tile 内执行 5 步
for kv in T.Pipelined(NUM_KV_BLOCKS, num_stages=3):
    qk(kv, K, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub)      # ① Cube: QK^T → L0C → UB
    softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, ...)                      # ② SimdVF: online softmax + NZ packing
    pack_p(P_nz_ub, P_shared)                                               # ③ L0C→UB NZ→ND 转换
    pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)   # ④ Cube: P@V → L0C → UB
    accumulate_output(O_tmp_ub, O_ub, alpha_ub)                             # ⑤ SimdVF: O = alpha*O + O_tmp
```

**NZ 布局 softmax packing（`core.py:91`）：**
- `T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})` 指定 NZ 布局
- `S.vcvt(e, "bfloat16", part=0/1)` 分离偶数/奇数位
- `S.vor(even, odd)` 合并为 bf16x128
- `S.vsstb(ptr, data, stride=((ROWS+1)<<16)|1, update=True)` NZ 布局存储
- `S.vexpdif(x, max)` 融合 exp(x-max) 避免 overflow

> **当前限制：** NZ packing 路径硬编码 `head_dim == 128`，`BR % 2 == 0`，`BC == 2 * VL (128)`。

### 3.5 模板 D：归约 / Norm 类算子

**核心：Fragment 寄存器复用 + Reducer 跨线程归约。**  
以 RMSNorm 为例（`example_rmsnorm.py`），关键技巧是**一次 UB→Fragment 加载，复用于 sum(x²) 和 y=x*rstd*w**，避免二次 UB 读。

```python
with T.SimtVF(threads=256):
    x_frag = T.alloc_fragment((TILE,), "float32")      # ① Fragment 寄存器
    for i in T.Parallel(TILE):
        x_frag[i] = x_ub[i]                              # ② UB→Fragment（一次）

    sum_sq = T.alloc_reducer((1,), "float32", op="sum", replication="all")
    T.clear(sum_sq)
    for i in T.Parallel(TILE):
        sum_sq[0] += x_frag[i] * x_frag[i]              # ③ 从寄存器归约（无 UB 读）
    T.finalize_reducer(sum_sq)                           # ④ 跨线程归约

    rstd = T.rsqrt(sum_sq[0] / d + eps)
    for i in T.Parallel(TILE):
        y_ub[i] = x_frag[i] * rstd * w_ub[i]             # ⑤ 复用 x_frag（无 x reload）
```

**性能（fp32, batch=4096）：** d=4096: ~100 us, 1300+ GB/s (80%+ peak)；d=7168: ~165 us, 1400+ GB/s (88%+ peak)

### 3.6 模板 E：量化 / Cast 类算子

**核心：SimdVF + simd intrinsics 实现 amax 归约 + FP8 转换。**  
以 per-token FP8 cast 为例（`example_simdvf_per_token_cast_to_fp8.py`）：

```python
with T.SimdVF():
    eps = S.vdup(1e-4, "float32")
    fp8_max_reg = S.vdup(448.0, "float32")
    for i in range(blk_m):
        for j in range(group_block):
            x0 = S.vld(y_ub[i, col])                     # ① 加载 64 个 float32
            x1 = S.vld(y_ub[i, col + 64])
            amax = S.vcmax(S.vmax(S.vabs(x0), S.vabs(x1)))  # ② 跨 lane amax 归约
            scale = S.vdiv(S.vmax(amax, eps), fp8_max_reg)
            scale_brc = S.vdupv(scale)                    # ③ 广播标量
            q0 = S.vdiv(x0, scale_brc)
            q0_fp8 = S.vcvt(q0, "float8_e4m3fn")          # ④ FP8 转换
            S.vsts(y_q_ub_fp8[i, col], q0_fp8, dist="PK4_B32")  # ⑤ PK4 布局存储
```

**关键：** `group_block` 自适应（64/32/16/8），`T.annotate_buffer_versions` 多缓冲流水。

### 3.7 模板 F：TopK / 选择类算子

**核心：SimdVF 选择排序循环，k 次迭代每次选出最大值。**  
以 MoE topk gate 为例（`example_simdvf_topk_gate.py`）：

```python
with T.SimdVF():
    v = [S.vld(s_ub[i * VL]) for i in range(4)]           # 4 个 vreg 覆盖 256 lane
    r = [S.vci(i * VL) for i in range(4)]                  # 对应 index
    for k in range(num_topk):
        mx = S.vdupv(S.vcmax(S.vmax(S.vmax(v[0],v[1]), S.vmax(v[2],v[3]))))  # ① 全局 max
        idx = S.vdupv(S.vcmin(...))                         # ② 最小 index（tie-break）
        S.vsts(out_ub[k], idx, one, "ONEPT_B32")            # ③ 存 winner
        for i in range(4):
            v[i] = S.vsel(neg_inf, v[i], S.vcmp(r[i], idx, "eq"))  # ④ mask out winner
```

**关键：** `T.copy(..., pad_value=-inf)` 处理非 32B 对齐尾部；`NUM_STAGES=6` 深流水。

### 3.8 通用设计决策清单

| 决策项 | 推荐选择 | 依据 |
|--------|----------|------|
| 核数（Vector 算子） | `get_num_vec_cores()`（= 64） | 纯 vector 算子用 vector 核 |
| 核数（GEMM 算子） | 32（cube_cores） | GEMM 受 Cube 单元限制 |
| 核间调度 | `T.Persistent` + `group_size=1` | 工作窃取，自动负载均衡 |
| UB 分配 | `T.alloc_shared`，≤ 256 KB/core | Ascend UB 上限 |
| 流水级数 | `num_stages=2~6` | GEMM 用 2，Vector/TopK 用 4~6 |
| 不规则 gather | ✅ 逐行 `T.copy` DMA | MTE2 引擎，绕过 L2 cache |
| 元数据访问 | ✅ 批量 `T.copy` 预加载到 UB | 一次 DMA 搬完小数组 |
| 边界处理 | `T.fill(values, 0)` + `if` 守卫 + `pad_value` | 非法 index 填零 |
| PassConfig | `TIR_DISABLE_VECTORIZE` + `TL_DISABLE_DATA_RACE_CHECK` | NPU 不需要向量化 lower |
| 多 dtype 支持 | `x.view(torch.int8)` 按字节处理 | 统一 kernel 适配多 dtype |
| 设备分发 | `vllm.platform` 打桩 + fallback | try import + torch.npu 检测 |
| Profiling | `do_bench(fn, backend="msprof")` | NPU 专用后端 |

---

## 四、高级优化技术

### 4.1 多缓冲（Double/Multi Buffering）

NPU 有**两种多缓冲机制**，可共存：

| 类型 | API | 原理 | 适用场景 | 示例 |
|------|-----|------|----------|------|
| 自动多缓冲 | `T.annotate_buffer_versions({buf: N})` | 编译器自动创建 N 个版本，<br>MTE2 读版本 i+1 时计算版本 i | 标准流水（读→算→写） | `example_rmsnorm.py`<br>`example_compress.py` |
| 手动多缓冲 | `T.annotate_manual_multi_buffer(buf)` | 用户手写 `buf[s % 2]` 索引，<br>编译器分析后插入 set_flag/wait_flag | loop-carried RAW + 并发 drain<br>（一步读 A 写 B，下一步读 B 写 A） | `example_manual_multibuffer.py` |

```python
# 自动多缓冲
T.annotate_buffer_versions({s_ub: 6, out_ub: 6})
for w in T.Pipelined(N, num_stages=6, annotations={"multi_buffer_eligible": [s_ub]}):
    T.copy(scores[row, :], s_ub[:num_experts])  # MTE2 版本 i+1
    with T.SimdVF():                              # Vector 版本 i
        ...                                        # 计算用 s_ub
    T.copy(out_ub[:k], topk_idx[row, :])          # MTE3 版本 i-1

# 手动多缓冲（ping-pong ring）
state_ub = T.alloc_shared((NUM_STAGES, width), dtype)
T.annotate_manual_multi_buffer(state_ub)
for s in T.Pipelined(num_steps, num_stages=NUM_STAGES):
    read_slot = s % NUM_STAGES
    write_slot = (s + 1) % NUM_STAGES
    state_ub[write_slot, k] = state_ub[read_slot, k] + delta_ub[k]  # 读 A 写 B
    T.copy(state_ub[write_slot, :], out[row, s, :])                  # MTE3 drain B
```

### 4.2 Split-K GEMM + 核间同步

当输出 tile 数不足以占满所有 AIC 时，沿 K 维切分，多核并行计算不同 K 分区，最后原子累加。

```python
with T.Kernel(NUM_BLOCKS) as bx:
    split_id = bx % split_k
    group_id = bx // split_k
    if split_id != 0:
        T.set_atomic("add", "float32")             # 非 split_0 用原子累加
    for out_tile in T.Serial(TILES_PER_GROUP):
        for local_kt in T.Pipelined(K_TILES_PER_SPLIT, num_stages=2):
            T.copy(X[...], x_l1); T.copy(W[...], w_l1)
            T.gemm(x_l1, w_l1, res, clear_accum=(local_kt == 0))
        with T.PerCoreTask():
            if split_id == 0:
                T.copy(res, C[...])                 # split_0 初始化
            T.ascend_sync_inter_arrive("PIPE_FIX", 0)  # 核间屏障
            T.ascend_sync_inter_wait("PIPE_FIX", 0)
            if split_id != 0:
                T.copy(res, C[...])                 # 其他 split 原子加
```

**确定性模式：** 逐 split 分阶段提交 + 每阶段后屏障，固定 FP32 归约顺序。

### 4.3 While 循环自动流水

AutoSchedule 无法直接流水 `while` 循环。TileLang 通过 `NormalizeControlFlowForSchedule` 将 while 重写为有界 for，流水后再恢复为 `while(true)` + `if (...) T.loop_break()`。

```python
for o in range(OUTER):
    i = 0
    while i < MID:                                   # while #1
        for j in T.Pipelined(SUB, num_stages=4):     # 可流水的内层 for
            T.copy(A[begin:end], b_ub)
            with T.SimtVF(threads=256):
                for k in T.Parallel(TILE):
                    b_ub[k] = b_ub[k] + bias
            T.copy(b_ub, C[begin:end])
        i = i + 1
```

### 4.4 动态形状与 StridedTensor

推理场景中 seq_len 等维度运行时变化，需用动态形状：

```python
total_q = T.dynamic("total_q")
score_stride0 = T.dynamic("score_stride0", dtype="int64")

@T.prim_func
def kernel(
    score: T.StridedTensor(shape=[total_q, OVERLAP, DIM],
                           strides=[score_stride0, DIM, 1], dtype=T.float32),
    ...
):
    T.assume(score_stride0 % DIM == 0)                    # 编译器提示
    T.assume_no_conflict(state_cache[...], cross=True)    # 跨迭代无冲突
    ...
```

### 4.5 L2 Cache 旁路

对于大粒度 DMA（≥ 128B），可通过旁路 L2 cache 减少缓存污染：
- `example_gemm_bypass_l2.py`：GEMM 旁路 L2
- Gather 优化中，批量 T.copy 天然绕过 L2（MTE2 直达 UB）

---

## 五、已有 NPU 算子的性能优化策略

### 5.1 瓶颈驱动优化流程（Bottleneck-Driven）

**核心原则：不盲目调参，先定位瓶颈再针对性优化。**

```
msprof 采集 → 识别瓶颈引擎 → 查表选策略 → 实施 → 验证闭环
```

### 5.2 瓶颈类型与优化措施对照表

| 瓶颈类型 | msprof 指标 | 优化措施 | 实际案例与效果 |
|----------|-------------|----------|----------------|
| ❌ Vector 标量读瓶颈 | aiv_vec_ratio > 90%<br>aiv_vec_wait_ratio > 95%<br>L2 read_hit_rate < 50% | ✅ 替换为 T.copy DMA<br>用 MTE2 引擎绕过 L2<br>增大单次传输粒度（1B→128B） | gather d=128: SimtVF→T.copy<br>296→87 us（3.4x）<br>带宽 11%→39% |
| ❌ Scalar 循环开销瓶颈 | aiv_scalar_ratio > 90% | ✅ 批量化处理（BM 行一批）<br>批量 T.copy 加载索引<br>减少迭代次数 N/BM 倍 | gather d=656: 逐行→批量(BM=4)<br>745→449 us（1.7x） |
| ⚠️ MTE2 带宽极限 | aiv_mte2_ratio > 95%<br>带宽利用率 > 60% | ⚠️ 判定接近硬件极限<br>尝试增大 BD / num_stages<br>若 <1% 改善则停止 | gather d=14336: 72% 带宽<br>6 种方案均无改善 |
| MTE2/MTE3 未重叠 | vec+scalar+mte2+mte3 ≈ 100%<br>（完全串行） | ✅ 启用 num_stages=4<br>四缓冲流水<br>MTE2 读与 MTE3 写重叠 | 所有 DMA 分支标配<br>num_stages=4 |
| 核间负载不均衡 | 部分核 scalar_ratio 异常高<br>Task Duration 方差大 | ✅ T.Persistent group_size=1<br>工作窃取自动均衡 | prepare_llama_decode<br>balance_sequences |
| 元数据反复 GM 访问 | scalar_time 含大量小数组 GM 读 | ✅ 批量 T.copy 预加载到 UB<br>一次 DMA 搬完 | gather_*_zos 系列<br>seqlens/cu_seqlens |
| 嵌套串行循环 | serial 循环内含 GM 访问 | ✅ SimtVF 展平到 1D 线程<br>每线程 1 元素 | gather_*_zos 系列<br>(seq,j,r) → 1D tx |
| GEMM Cube 未饱和 | aic_mad_ratio < 80% | ✅ 增大 TILE_K<br>调整 num_stages<br>aswt_swizzle 提升 L2 | example_gemm.py<br>TILE_K=256 (bf16) / 128 (fp32) |
| FixPipe 与 GEMM 未重叠 | aic_fixpipe_ratio 高<br>且与 MAD 串行 | ✅ unit_flag_ctrl 重叠<br>UF_2 (中间帧) / UF_3 (尾帧) | example_gemm.py<br>unit_flag_ctrl=T.Select(...) |

### 5.3 带宽利用率基准

| 带宽利用率 | 评价 | 行动 |
|------------|------|------|
| ✅ > 60% | 接近硬件极限 | 停止优化，记录已达极限 |
| ⚠️ 30% ~ 60% | 有优化空间 | 检查是否 Vector/Scalar 瓶颈阻塞了 DMA |
| ❌ < 30% | 严重未利用 | 必然存在非 DMA 瓶颈，优先消除 |

**实测数据（Ascend 950DT，峰值 4000 GB/s）：**

| 场景 | 优化前 | 优化后 | GPU 带宽 |
|------|--------|--------|----------|
| gather d=128 | ❌ 455 GB/s (11%) | ✅ 1544 GB/s (39%) | 5926 GB/s |
| gather d=656 | ❌ 924 GB/s (23%) | ✅ 1532 GB/s (38%) | 5727 GB/s |
| gather d=14336 | ✅ 2858 GB/s (72%) | ✅ 2858 GB/s (72%) | 6573 GB/s |
| RMSNorm d=4096 | — | ✅ 1300+ GB/s (80%+) | — |
| RMSNorm d=7168 | — | ✅ 1400+ GB/s (88%+) | — |

---

## 六、Agent 自动化算子开发与优化闭环方案

集成 ops-profiling skill 的 6 步工作流（采集 → 归档 → 判定 → 定位 → 优化 → 验证），  
结合全量 examples 的六大算子模板和 msprof 8-CSV 分析体系，构建端到端自动化闭环。

### 6.1 闭环架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      Agent 自动优化闭环（10 步）                           │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ ① 分析   │→ │ ② 编写   │→ │ ③ 正确性 │→ │ ④ 准备   │→ │ ⑤ msprof │  │
│  │ 算子规格 │  │ 初始版本 │  │ 验证     │  │ profiling│  │ op 采集  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│       ↑                                                        │        │
│       │              ┌──────────┐         ┌──────────┐         │        │
│       │              │ ⑩ 报告   │←──┌──── │ ⑧ 瓶颈   │←──┐    │        │
│       │              └──────────┘   │     │ 定位优化 │   │    │        │
│       │                             │     └──────────┘   │    │        │
│       │              ┌──────────┐   │                    │    │        │
│       └──────────────│ ⑨ 验证   │←──┘  ┌──────────┐     │    │        │
│                      │ 优化效果 │        │ ⑦ 性能   │←────┘    │        │
│                      └──────────┘        │ 标准判定 │          │        │
│                                          └──────────┘          │        │
│                               ┌──────────┐                    │        │
│                               │ ⑥ 归档   │←───────────────────┘        │
│                               │ + 摘要   │                              │
│                               └──────────┘                              │
└──────────────────────────────────────────────────────────────────────────┘
```

**与 ops-profiling skill 的映射：** 步骤 ④⑤⑥ 对应 skill Step 1~3（准备→采集→归档），步骤 ⑦ 对应 skill Step 4（判定），步骤 ⑧ 对应 skill Step 5（定位优化），步骤 ⑨ 对应 skill Step 6（验证）。

### 6.2 十步详细流程

#### Step ①：分析算子规格

| 输入 | 内容 | Agent 动作 |
|------|------|------------|
| 算子语义 | Python 参考实现 / PyTorch 等价操作 / 数学公式 | 理解计算逻辑 |
| 输入输出 | shape、dtype、device（npu/cuda） | 确定 buffer 类型 |
| 关键维度 | D 的典型取值范围（如 d=4/128/656/14336） | 规划分支策略 |
| 访问模式 | 连续 / 不规则 gather / GEMM / 归约 / 量化 | 选择模板 A~F |
| GPU 参考实现 | 已有的 `_gather_tl` 等 | 正确性对照 + 性能基线 |
| 算子类型判定 | Elementwise / Reduction / MatMul / 搬运 / 混合 | 预期 ratio 分布（见 §6.5） |

**输出：** 算子类型标签（A~F）、预期主导流水（VEC/CUBE/MTE2）、分支策略。

#### Step ②：编写初始版本

**Agent 动作：** 按 §3 对应模板生成代码，自动完成：
- 设置 `pass_configs`（`TIR_DISABLE_VECTORIZE` + `TL_DISABLE_DATA_RACE_CHECK` + `TL_ENABLE_FAST_MATH`）
- 计算 tiling 参数（TILE_M/N/K / BD / BM / num_cores）
- 根据算子类型选择执行模型（SimdVF / SimtVF / Cube）
- 添加边界保护（copy_m / copy_d / T.fill / pad_value）
- 编写入口函数（view int8 + 设备分发 + vllm.platform 打桩）
- 编写 pytest 正确性测试（对照 GPU/CPU reference）和 benchmark
- **编写 profiling 入口脚本**（内嵌 `cases.csv`，`--case-id` 逐场景运行，显式调用目标 kernel）

#### Step ③：正确性验证

```bash
# 本地 → 远程同步
python ssh_remote.py sync

# 远程正确性测试
python ssh_remote.py exec "cd <remote_dir> && python -m pytest <test_file> -k test_<kernel> -x -v"
```

**判定：** 所有 parametrize case PASSED。若失败，Agent 分析错误信息，修正 kernel 逻辑后重试。

#### Step ④：准备并校验 profiling 入口（ops-profiling Step 1）

从当前源码**重新生成**独立 `profiling_file`（禁止复用旧文件测量新 kernel）。该文件必须：
- 内嵌最终 `cases.csv`（每个 case 一行：shape、dtype、case_id）
- 通过 `--case-id` 一次只运行一个 case
- 显式调用文件内的本地目标 kernel（非通过第三方框架间接调用）

采集前先触发 TileLang JIT 并确认目标 kernel 在 NPU 上实际运行：

```bash
python <profiling_file> --case-id <case_id>
```

**校验要点：** 确认输出的 kernel name，后续 msprof 的 `--kernel-name` 必须与此精确匹配。

#### Step ⑤：msprof op 采集（ops-profiling Step 2）

对 `cases.csv` 中每个 case 启动**两次独立采集**：

```bash
# ① 普通采集（用于计时和 ratio 分析）
msprof op --warm-up=10 --launch-count=5 --output=<case_output> \
  --kernel-name=<expected_kernel> \
  python <profiling_file> --case-id <case_id>

# ② PipeTimeline 采集（用于流水时序和气泡分析，独立目录，不用于计时）
msprof op --warm-up=10 --launch-count=1 --replay-mode=kernel \
  --aic-metrics=PipeTimeline --output=<case_output>_pipe_timeline \
  --kernel-name=<expected_kernel> \
  python <profiling_file> --case-id <case_id>
```

| 参数 | 说明 | 何时使用 |
|------|------|----------|
| `--warm-up=10` | 预热 10 次后采集 | **始终使用**，避免 DVFS 影响 |
| `--launch-count=5` | 运行 5 次取均值 | 需要统计稳定性时 |
| `--kernel-name` | 只采集目标 kernel | **必须使用**，防止辅助 kernel 混入 |
| `--aic-metrics=PipeTimeline` | 流水时序和气泡 | 独立目录、`launch-count=1`，不用于计时 |

**输出：** 在指定目录下生成 `OPPROF_{timestamp}_XXX/` 文件夹，包含 8 个 CSV 文件。

#### Step ⑥：归档数据 + 生成统计摘要（ops-profiling Step 3）

```bash
# 找到最新 OPPROF 目录
OPPROF_DIR=$(ls -td <output_dir>/OPPROF_* | head -1)

# 归档 CSV + 生成摘要
python3 {skill_path}/scripts/perf_summary.py $OPPROF_DIR <variant_output_dir> \
  --kernel-name <expected_kernel> --round-name <case_or_round_name>

# PipeTimeline 完整保留
TIMELINE_DIR=$(ls -td <case_output>_pipe_timeline/OPPROF_* | head -1)
cp -a "$TIMELINE_DIR" <variant_output_dir>/docs/perf/<case_or_round_name>/pipe_timeline
```

`perf_summary.py` 自动完成：
1. 在 `<variant_output_dir>/docs/perf/<round_name>/` 创建归档目录
2. 复制全部 8 个 CSV 原始文件
3. 校验 `<expected_kernel>` 在 `OpBasicInfo.csv` 中**唯一匹配**（零匹配或多匹配立即失败）
4. 生成 `summary.txt` 统计摘要（AIC/AIV 混合核分别统计，**不做判定**）

**归档目录结构：**
```
<variant_output_dir>/docs/perf/
├── round_001/                    # 基线
│   ├── OpBasicInfo.csv           # 8 个原始 CSV
│   ├── PipeUtilization.csv       # 最重要：各流水线单元耗时和占比
│   ├── ArithmeticUtilization.csv # Cube/Vector 指令 cycle 占比和计算量
│   ├── Memory.csv                # 内存读写带宽和搬运量
│   ├── MemoryL0.csv              # L0A/L0B/L0C 带宽
│   ├── MemoryUB.csv              # UB 读写带宽
│   ├── L2Cache.csv               # L2 命中率
│   ├── ResourceConflictRatio.csv # Bank conflict 和资源冲突
│   ├── summary.txt               # 统计摘要（min/avg/max）
│   └── pipe_timeline/            # PipeTimeline 时序数据
├── round_002/                    # 优化后
│   └── ...
```

#### Step ⑦：性能标准判定（ops-profiling Step 4）

**Agent 分析流程（7 步）：**
1. **读 `summary.txt`** — 获取全局概览（~30 行紧凑文本）
2. **结合 `csv_fields_reference.md`** — 理解各指标含义和阈值
3. **发现异常时读原始 CSV** — 如核间不均衡，Read `PipeUtilization.csv` 查看逐核数据
4. **读 `pipe_timeline/`** — 分析关键核流水先后、重叠、等待和气泡
5. **核算搬运量与计算量** — 从源码枚举 GM 读写张量、Cube/Vector 计算量
6. **结合 `optimization_quickref.md`** — 将瓶颈映射为 TileLang 修改方向
7. **用中文输出分析文件**

**7a. 总体判定流程**
```
读取 OpBasicInfo.csv → 获取 Task Duration 和 Block Dim
    ↓
读取 PipeUtilization.csv → 找到各流水占比最高的单元
    ↓
核算实际搬运数据量 + 实际运算数据量（区分 Cube/Vector）
    ↓
计算理论耗时（搬运量/带宽 或 计算量/算力）
    ↓
比较实际耗时 vs 理论耗时
    ├── 差距 <20% → 性能达标（接近硬件极限）
    ├── 差距 20-50% → 有优化空间，查阅瓶颈优化表
    └── 差距 >50% → 严重瓶颈，必须优化
```

**7b. 各指标达标标准**

| 指标 | 达标 | 警告 | 严重问题 |
|------|------|------|----------|
| 核间负载均衡 | ✅ 各核差异 <10% | ⚠️ 10-30% | ❌ >30% |
| Block Dim | ✅ 等于可用核数 | ⚠️ 远小于核数 | ❌ = 1 |
| VEC ratio | ✅ 与算子类型匹配 | ⚠️ >80% | ❌ >90% 且无优化空间 |
| MTE2 ratio | ✅ <30%（计算型） | ⚠️ 30-50% | ❌ >50% |
| fixpipe_ratio | ✅ <5% | ⚠️ 5-15% | ❌ >15% |
| icache_miss_rate | ✅ <5% | ⚠️ 5-15% | ❌ >15% |
| bank conflict | ✅ <5% | ⚠️ 5-15% | ❌ >15% |
| L2 Cache 命中率 | ✅ >80% | ⚠️ 50-80% | ❌ <50% |
| 头开销占比 | ✅ <10% | ⚠️ 10-30% | ❌ >30% |
| DoubleBuffer 重叠 | ✅ MTE2/VEC 重叠 >30% | ⚠️ 10-30% | ❌ <5% |
| 带宽利用率 | ✅ >60% | ⚠️ 30-60% | ❌ <30% |

**7c. 不同算子类型的预期 ratio 分布**

| 算子类型 | 主导流水 | 预期 ratio | 异常信号 |
|----------|----------|------------|----------|
| Elementwise（Add/Mul/Relu） | VEC | vec_ratio 50-80% | MTE2 > VEC |
| Reduction（ReduceSum/Max） | VEC | vec_ratio 40-70% | scalar >20% |
| Activation（Softmax/Gelu） | VEC | vec_ratio 60-85% | 大量 cast |
| MatMul | CUBE | cube_ratio 40-70% | vec > cube |
| 纯搬运（Transpose/Concat） | MTE2/MTE3 | mte2+mte3 >50% | VEC >30% |

**7d. 理论耗时计算**

**前置：** 从算子源码核算实际搬运数据量与运算数据量（区分 Cube/Vector）。

```
# 搬运理论耗时
理论耗时(us) = 实际搬运数据量(Byte) / GM 峰值带宽
# Ascend950DT: GM 峰值带宽 ≈ 4 TB/s = 4000 GB/s

# 计算理论耗时
理论耗时(us) = 实际计算量(FLOP) / 对应单元理论算力
# Ascend950DT:
#   cube_fp16_bf16: 432 TFLOPS    cube_fp32: 26 TFLOPS    cube_int8_fp8: 864 TOPS
#   vector_fp32_add: 13.5 TFLOPS   vector_fp16_bf16_fma: 54 TFLOPS
```

**7e. 搬运量与计算量核算**

**搬运量（GM↔UB）：** 从源码枚举所有 GM 张量读写，逐个核算字节数。

| 方向 | 张量 | 字节计算 |
|------|------|----------|
| 读 | 每个输入 | 元素总数 × 单位元素字节数 |
| 写 | 每个输出 | 元素总数 × 单位元素字节数 |

**计算量（区分 Cube/Vector）：**
- **Cube：** `T.gemm` 路径 → 每 tile `2 × M × N × K` FLOP；无 gemm 则为 0
- **Vector：** `T.SimdVF` / `T.Parallel` 内逐元素运算，按「每元素运算次数 × 元素数」核算

```
有效带宽(GB/s)   = 总搬运字节 / 实测稳态耗时
带宽利用率       = 有效带宽 / GM 峰值带宽
实际算力(FLOP/s) = 实际计算量 / 实测稳态耗时
```

**7f. PipeTimeline 流水气泡分析**

只分析关键路径核及最慢/最快核，记录：
- Scalar、MTE2、VEC/CUBE、MTE3 的首尾时间、活跃时间和空闲区间
- MTE2↔计算、计算↔MTE3 的重叠时间与重叠率
- `SetFlag`/`WaitFlag`、barrier 等同步等待区间
- 首条有效流水前的头气泡、相邻指令簇间气泡、末条流水后的尾气泡

```
气泡率   = 流水空闲时间 / 关键路径时间
重叠率   = 两流水重叠时间 / 较短流水活跃时间
头尾开销 = Task Duration - 有效流水覆盖时间
```

**交付门禁：** 必须计算关键核的头/内部/尾气泡、总气泡率和关键流水重叠率，写入 `summary.txt` 与 `analysis.md`。

**流水重叠快捷判据：** `vec + scalar + mte2 + mte3` 四个 ratio 加起来 ≈100% 说明完全串行、该上双缓冲；加到 130% 以上才算真重叠。

#### Step ⑧：瓶颈定位与优化（ops-profiling Step 5）

确认瓶颈类型后，查阅 `optimization_quickref.md`，结合目标 TileLang 源码制定具体优化方法。

**8a. 瓶颈快速查找表**

| 瓶颈类型 | 判定条件 | 首选优化（TileLang 实现） | 实际案例 |
|----------|----------|---------------------------|----------|
| ❌ VEC Bound | `aiv_vec_ratio` 最高 | ① UB 融合（`T.alloc_shared` 保留中间值）<br>② 寄存器复用（`T.alloc_fragment`）<br>③ 减少 Cast（合并 dtype 转换）<br>④ 优化归约（`T.alloc_reducer`）<br>⑤ 调整 `T.SimtVF(threads=...)` | RMSNorm: fragment trick<br>消除 x 二次 UB 读 |
| ❌ MTE2/MTE3 Bound | `ai*_mte2_ratio` 最高 | ① 减少 GM 往返（融合计算）<br>② 增大连续搬运粒度（增大 Tile）<br>③ 处理非对齐尾块（`pad_value`）<br>④ 重用常量/权重（保留在 UB/L1）<br>⑤ 搬运计算重叠（`num_stages=N`）<br>⑥ 调整 L2 策略（`l2_cache_ctrl`） | gather d=128: SimtVF→T.copy<br>296→87 us（3.4x） |
| ❌ CUBE Bound | `aic_cube_ratio` 最高 | ① 调整矩阵 Tile（`TILE_M/N/K`）<br>② L1 数据复用（`T.alloc_l1`）<br>③ L0 累加（`T.alloc_l0c` + `clear_accum`）<br>④ K 流水（`T.Pipelined(K_TILES, num_stages=N)`）<br>⑤ 持久化调度（`T.Persistent`）<br>⑥ 输出路径（`T.copy` / `T.dual_copy`） | GEMM: TILE_K=256 (bf16)<br>aswt_swizzle 提升 L2 |
| ❌ SCALAR Bound | `ai*_scalar_ratio` >30% | ① 编译期特化（dtype/shape 移入 JIT 参数）<br>② 移出循环不变量<br>③ 减少 `T.serial`（改 `T.Parallel`）<br>④ 减少动态标量访问<br>⑤ 调整 block 数<br>⑥ 合并小 Tile | gather d=656: 逐行→批量(BM=4)<br>745→449 us（1.7x） |
| ⚠️ 核间不均衡 | 各核耗时差异 >10% | ① 检查 `T.Kernel(num_blocks)` 是否匹配 Tile 数<br>② `T.ceildiv` 计算 Tile 数<br>③ 尾块分散到多 block<br>④ `T.Persistent` 工作窃取 | prepare_llama_decode<br>balance_sequences |
| ⚠️ Bank Conflict | `aiv_vec_total_cflt_ratio` >5% | ① 调整 `T.alloc_shared` shape/padding<br>② 改变行 stride 和 `T.Parallel` 索引映射<br>③ 添加 padding 避免同周期命中同 bank | — |
| 流水重叠不足 | `vec+scalar+mte2+mte3` ≈100% | ① `T.annotate_buffer_versions({buf: N})` 自动多缓冲<br>② `annotations={"multi_buffer_eligible": [buf]}`<br>③ `T.annotate_manual_multi_buffer` 手动 ping-pong<br>④ 检查真实 RAW/WAR 依赖 | 所有 DMA 分支标配<br>num_stages=4 |
| L2 Cache 命中率低 | `ai*_total_hit_rate` <50% | ① Tile + 持久化调度增加局部性<br>② `T.copy(..., l2_cache_ctrl="NORMAL_FV")` 保留策略<br>③ `NOTALLOC_*` 流式访问策略<br>④ 对输入/权重/输出分别 A/B 测试 | gather: 批量 T.copy<br>天然绕过 L2 |
| 头开销大 | 头开销占比 >30% | ① 减少核数和动态分支<br>② 简化 kernel 启动路径<br>③ 合并小 Tile 增大每 block 工作量 | 小 shape 场景 |

**8b. 交叉关联诊断**

| 现象组合 | 根因假设 | TileLang 优先检查 |
|----------|----------|-------------------|
| 高 vec_ratio + 高 bank conflict | UB 布局放大 Vector 耗时 | `T.alloc_shared` shape/padding、并行索引映射 |
| 高 mte2_time + 低 L2 hit rate | 数据复用或 L2 策略不合理 | Tile 调度、`T.Persistent`、`l2_cache_ctrl` |
| 高 fixpipe_ratio | 输出路径或地址对齐低效 | 输出 Tile、有效区间、`T.dual_copy` 路径 |
| 高 mte2 + 高 mte3 | GM 双向搬运饱和 | 融合中间结果、增大 Tile、减少 GM 往返 |
| 低 Block Dim + 高 Duration | 并行 Tile 数或 block 设置不足 | `T.Kernel` block 数、Tile shape |
| scalar 高 + 小 shape | 动态控制和启动占比高 | 编译期特化、减少 `T.serial` |

**8c. 仓库实现参考**

优先从与目标算子结构接近的文件抽取模式：

| 优化模式 | 参考文件 |
|----------|----------|
| Vector Tile、UB 搬运、流水 | `vllm-ascend/softmax.py` |
| 动态 shape、UB/L1 流水 | `quant/cast_back_asc.py` |
| Buffer 多版本 | `vllm-ascend/cache.py` |
| `T.Pipelined` + `T.annotate_buffer_versions` | `ascend/example_simdvf_vecadd.py` |
| Fragment 复用和 Reduction | `ascend/example_rmsnorm.py` |
| `T.alloc_l1`/`T.alloc_l0c`/`T.gemm` | `ascend/example_gemm.py` |
| `T.copy(..., l2_cache_ctrl=...)` | `ascend/example_gemm_bypass_l2.py` |
| 自动与手动多缓冲 | `ascend/example_manual_multibuffer.py` |
| Cube+Vector 协作 + NZ 布局 | `ascend/flash_attention/core.py` |
| SimdVF 选择排序 + pad_value | `ascend/example_simdvf_topk_gate.py` |

**抽取优化模式时必须保留目标算子的接口、数据布局、边界处理和精度语义，不能整文件替换目标 kernel。**

#### Step ⑨：验证优化效果（ops-profiling Step 6）

每次优化后，重新运行 Step ⑤⑥。省略 `--round-name` 时数据自动归档为 `round_NNN+1`；显式命名时必须为新目录，脚本拒绝覆盖旧归档。

```bash
# 对比两轮摘要
diff <variant_output_dir>/docs/perf/round_001/summary.txt \
     <variant_output_dir>/docs/perf/round_002/summary.txt
```

**对比要点：**
1. Task Duration 是否下降
2. 瓶颈单元的 ratio 是否改善
3. 核间均衡是否改善（aiv_time min/max 差距）
4. 是否引入新的瓶颈

**每次只改变一个主要变量**，并使用相同 `cases.csv`、相同 kernel 过滤规则和相同计时口径重新采集。

**迭代终止条件：**
- 带宽利用率 > 60% **或**
- 实际耗时 vs 理论耗时差距 < 20% **或**
- 连续 3 次优化 < 5% 改善 **或**
- 编译失败（回退该参数）

#### Step ⑩：生成优化报告

自动生成 HTML 报告，包含：
- 各场景优化前后对比（耗时、加速比、带宽利用率、各引擎 ratio 变化）
- 与 GPU 性能对比（差距倍数）
- 使用的优化手段及效果（含 §6.6 瓶颈对照表映射）
- 瓶颈分析过程（msprof 8-CSV 指标变化、PipeTimeline 气泡分析）
- 理论耗时 vs 实际耗时对比（搬运量/计算量核算）
- 负结果记录（尝试过但无效的方案，如 d=14336 的 6 种尝试）
- 归档数据路径（`docs/perf/round_NNN/`）

### 6.3 msprof 8-CSV 联合诊断顺序

Agent 按**固定顺序**读取 8 个 CSV，逐步缩小瓶颈范围：

| 顺序 | CSV 文件 | 分析目标 | 关键字段 |
|------|----------|----------|----------|
| 1 | `OpBasicInfo.csv` | 校验 kernel 名、频率、Task Duration、Block Dim | Op Name, Task Duration(us), Block Dim, Current Freq |
| 2 | `PipeUtilization.csv` | 确定主导流水、kernel_time、核间差异 | aiv_vec_ratio, aic_cube_ratio, ai*_scalar_ratio, ai*_mte2_ratio, ai*_mte3_ratio |
| 3 | `ArithmeticUtilization.csv` | 判断有效计算、Cast、指令结构 | aic_cube_fops, aiv_vec_fp32_ratio, aiv_vec_fops |
| 4 | `Memory.csv` | 计算总搬运量、带宽利用率、理论搬运时间 | read_main_memory_datas, GM_to_UB_datas, GM_to_UB_bw_usage_rate |
| 5 | `MemoryL0.csv` | CUBE kernel：L0A/L0B/L0C 带宽 | aic_l0a_read_bw, aic_l0c_write_bw_cube |
| 6 | `MemoryUB.csv` | Vector kernel：UB 读写带宽 | aiv_ub_read_bw_vector, aiv_ub_read_bw_scalar |
| 7 | `L2Cache.csv` | 验证缓存命中假设 | ai*_total_hit_rate, ai*_read_hit_rate |
| 8 | `ResourceConflictRatio.csv` | Bank conflict、资源冲突、等待比例 | aiv_vec_total_cflt_ratio, aiv_vec_wait_ratio, aic_cube_wait_ratio |

**第 9 步：** 将结论映射到 `T.Kernel`、Tile shape、`T.copy`、buffer scope、`T.Pipelined`、`T.Persistent`、`T.Parallel` 或 `T.gemm` 的具体修改。

### 6.4 上板 vs 仿真选择

| 维度 | 上板 (msprof op) | 仿真 (msprof op simulator) |
|------|------------------|----------------------------|
| 需要 NPU | 是 | 否 |
| 时序精度 | 真实硬件时序 | 周期级模型估算 |
| 输出 | 8 个 CSV 文件 | CSV + trace.json |
| 资源冲突数据 | ✅ 有 | ❌ 无 |
| L2 Cache | ✅ 真实命中率 | 估算 |
| DVFS 影响 | 有（需 warm-up） | 无 |
| 适合阶段 | 性能验收、生产调优 | 早期开发、指令级调试 |

**建议：** 开发阶段用仿真快速迭代，验收阶段用上板确认真实性能。

### 6.5 注意事项

1. **必须 warm-up：** 首次运行受 DVFS 影响，耗时偏高。始终使用 `--warm-up=10`
2. **频率检查：** 读取 `OpBasicInfo.csv` 的 `Current Freq` 和 `Rated Freq`，若 Current < Rated 说明芯片未满频
3. **MTE2/MTE3 带宽共享：** 同时读写 GM 时，总带宽被共享，理论耗时应按 `(MTE2搬运量 + MTE3搬运量) / GM带宽` 计算
4. **PipeTimeline 不计时：** 动态插桩会改变耗时，只用于流水先后、重叠和气泡分析
5. **小数据量场景：** 数据量很小时头开销占比会很高，这不一定是算子问题，而是数据量不足
6. **多核同地址访问：** 多核同时读同一 512B 地址范围会被串行化，导致 MTE2 耗时异常
7. **每次只改一个变量：** 修改 Tile/buffer/stage 后先跑精度回归，再重新采集全部 case
8. **UB/L1/L0 容量检查：** Tile 大小、buffer 数量和 pipeline stage 必须满足目标设备容量；编译失败或资源超限时回退

### 6.6 Agent Prompt 模板

```
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
⑧ 按 §6.2 Step⑧ 瓶颈对照表选择优化策略，结合源码制定修改方案
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

### 6.7 所需工具链

| 工具 | 用途 | 关键参数/路径 |
|------|------|---------------|
| `ssh_remote.py` | 本地→远程同步、测试、执行 | sync / run / bench / exec |
| `msprof op` | NPU 性能采集 | `--kernel-name`（必须）、`--warm-up=10`、`--launch-count=5` |
| `perf_summary.py` | 归档 CSV + 生成摘要 | `--kernel-name`、`--round-name` |
| `profiling_file.py` | profiling 入口脚本 | `--case-id` 逐场景，内嵌 cases.csv |
| `tile_kernels.config` | 核数查询 | `get_num_vec_cores()` |
| `pytest` | 正确性 + benchmark | `--run-benchmark` |
| `csv_fields_reference.md` | 8 个 CSV 字段定义和阈值 | Step ⑦ 分析时查阅 |
| `optimization_quickref.md` | 瓶颈→TileLang 优化映射 | Step ⑧ 定位瓶颈后查阅 |
| ops-profiling skill | 完整 msprof 分析工作流 | 6 步：采集→归档→判定→定位→优化→验证 |

---

## 七、核心经验总结

### 写 NPU 算子的五条黄金法则

1. **DMA 优先：** 所有 GM 访问用 `T.copy`（MTE2/MTE3 引擎），**绝不**用 SimtVF 标量读 GM。唯一例外：数据量 < 1KB。
2. **批量摊薄：** 索引加载、输出写入都要批量化（BM 行一批），减少 scalar 循环开销和 DMA 次数。
3. **流水重叠：** `num_stages=2~6` 多缓冲，让 MTE2 读和 MTE3 写并行重叠。检查各引擎 ratio 之和 > 100% 确认流水生效。
4. **引擎匹配：** GEMM 用 Cube（L1/L0），向量计算用 Vector（UB/Fragment），两者通过 `T.dual_copy` 衔接。不要在 Vector 上做 GEMM 或在 Cube 上做向量计算。
5. **寄存器复用：** 用 `T.alloc_fragment` 缓存中间结果（如 RMSNorm 的 x_frag），避免重复 UB 加载。归约用 `T.alloc_reducer + T.finalize_reducer`。

### 优化 NPU 算子的四个陷阱

1. **盲目调参：** 不先 msprof 定位瓶颈就调 num_stages / BD / group_size，大概率无效。先定位瓶颈引擎再针对性优化。
2. **忽视 Scalar 开销：** NPU 的 Scalar 循环控制 + 索引加载开销可达 99.7%，必须通过批量化摊薄。
3. **过度优化已达极限的场景：** 带宽利用率 > 60% 时，多种方案可能都 < 1% 改善。学会判定硬件极限。
4. **忽视 NZ 布局：** Flash Attention 中 softmax 输出必须以 NZ 布局存储，否则第二次 GEMM 需额外转换。用 `make_ascend_compact_nz_layout` + `S.vsstb`。

### Agent 自动化的关键成功因素

- **瓶颈对照表是核心知识：** Agent 靠 "msprof 指标 → 优化策略" 映射表自动决策
- **验证闭环不可省略：** 每次修改必须 sync → 正确性 → benchmark
- **负结果同样重要：** 记录无效方案，避免 Agent 重复探索
- **远程执行闭环：** 本地开发 + SSH 远程 NPU 验证
- **代码长度约束：** NPU kernel 不应超过 GPU kernel 的 2 倍行数

---

## 八、附录：examples 目录算子分类索引

### 8.1 Ascend NPU 专用示例（`examples/ascend/`）

| 类别 | 文件 | 关键技术 |
|------|------|----------|
| **GEMM** | `example_gemm.py` | auto-schedule, aswt_swizzle, unit_flag_ctrl, dual_copy, hf32 |
| | `example_gemm_l0.py` | 显式 L0A/L0B/L0C, L1→L0 子循环 |
| | `example_gemm_bypass_l2.py` | L2 旁路 |
| | `example_gemm_splitk.py` | Split-K, inter-core sync, atomic, deterministic |
| | `example_gemm_mixedkernel.py` | 混合 kernel |
| | `example_gemm_ub_merge.py` | UB merge |
| | `example_gemm_various_shapes.py` | 多形状测试 |
| | `example_blockscaled_gemm.py` | MXFP8/MXFP4 block-scaled, scale factors |
| | `example_blockscaled_gemm_l0.py` | 显式 L0 block-scaled |
| **Flash Attn** | `flash_attention/core.py` | Cube+Vector, NZ layout, SIMD softmax packing |
| | `flash_attention/example_mha.py / gqa.py` | MHA/GQA 前向 |
| **Norm** | `example_rmsnorm.py` | Fragment trick, reducer, double buffer, 64 cores |
| **Atomic** | `example_atomic.py` | GM atomic add/max/min |
| **Compress** | `example_compress.py` | 两阶段 compress + state-cache, dynamic, StridedTensor |
| **量化** | `example_simdvf_per_token_cast_to_fp8.py` | SimdVF, amax 归约, vcvt FP8 |
| | `example_simtvf_per_token_cast_to_fp8.py` | SimtVF 版本 |
| **TopK** | `example_simdvf_topk_gate.py` | MoE topk gate, 选择排序, pad_value |
| | `example_simdvf_scalar_topk.py` | 标量 topk |
| **VecAdd** | `example_simdvf_vecadd.py` | SimdVF 纯向量 |
| | `example_simtvf_vecadd.py` | SimtVF SIMT 向量 |
| | `example_simtvf_vecadd_mutex.py` | SimtVF + mutex 同步 |
| | `example_vmi_vecadd.py` | VMI 执行模型 |
| **高级调度** | `example_while_pipelined.py` | while 循环自动流水 |
| | `example_manual_multibuffer.py` | 手动 ping-pong 多缓冲 |
| | `example_crosslevel_multibuffer.py` | 跨层多缓冲 |
| | `example_buffer_version_annotation.py` | 缓冲版本标注 |

### 8.2 TileKernels-Nightly 算子库

| 模块 | 代表文件 | 说明 |
|------|----------|------|
| **vllm-ascend** | `gather_opt.py` | Gather 优化版（本文核心实践） |
| | `rope.py` | RoPE 旋转位置编码 |
| | `softmax.py` | Softmax / LogSoftmax |
| | `norm.py` | RMSNorm/LayerNorm（含 residual/bias/fp8） |
| **quant** | `per_token_cast_*_asc.py` | per-token FP8 cast（含随机舍入） |
| | `per_block_cast_*_asc.py` | per-block cast（最复杂，48K 行） |
| | `swiglu_forward_*_asc.py` | SwiGLU 前向 |
| **moe** | `moe_topk_gate_*_asc.py` | MoE topk gate（50K 行） |
| **mhc** | `norm_fn_*_asc.py` | MHC norm（33K 行） |
| | `sinkhorn_*_asc.py` | Sinkhorn 归一化 |
| **engram** | `engram_gate_*_asc.py` | Engram gate（27K 行） |
| **transpose** | `batched_transpose_asc.py` | 批转置 |

### 8.3 DeepSeek 系列内核

| 模块 | 关键算子 | 说明 |
|------|----------|------|
| deepseek_mla | MLA decode (split/paged/persistent/ws) | Multilinear Attention |
| deepseek_nsa | NSA fwd/decode/bwd (varlen) | Native Sparse Attention |
| deepseek_v32 | sparse MLA, topk selector, FP8 indexer | DeepSeek V32 推理 |
| deepseek_v4 | FP8-FP4 GEMM, sparse attn, act_quant | DeepSeek V4 (Blackwell) |
| deepseek_mhc | MHC pre/post/bwd | Multi-Head Context |
| dsa_hisa | HISA indexer, FP8 pooling, block sparse MQA | DSA/HISA 索引 |

---

*基于 Ascend 950DT 全量 examples 目录归纳总结*  
*涵盖 60+ 子目录、200+ Python 文件、六大算子类型*  
*tilelang + msprof + simd intrinsics | 2026-08-26*
