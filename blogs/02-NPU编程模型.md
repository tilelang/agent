# 第 2 篇：TileLang NPU 编程模型

> **定位**：API 工具箱详解（参考手册定位）。读完本篇，读者能写出基本的 TileLang NPU kernel。
>
> **知识边界**：
> - ✅ 本篇讲：API 签名、参数含义、基本用法、代码示例
> - ❌ 本篇不讲：策略选择/何时用（→ 第 5 篇）、Pass 原理（→ 第 3 篇）、完整算子结构（→ 第 4 篇）
>
> **前置**：第 1 篇（了解方法论概念）　**后续**：第 3 篇解释这些 API 如何编译到硬件

---

## 2.1 TileLang 介绍与架构

### 2.1.1 定位

TileLang 是基于 TVM 的 **Tile-Level DSL**（领域特定语言），核心设计理念是让开发者只需描述 **tile 级计算**（数据搬运 + tile 计算 + 写回），后端自动 lowering 到具体硬件指令。

```
开发者视角（TileLang DSL）          编译器视角（TVM IR + Pass）
┌─────────────────────────┐       ┌─────────────────────────┐
│ @tilelang.jit            │       │ Python AST → TIR         │
│ def kernel(M, N, K):     │  ──→  │ → 通用 Pass (Simplify...) │
│     with T.Kernel():     │       │ → 后端 Pass (Ascend/CUDA) │
│         shared = alloc() │       │ → Codegen (bisheng/nvcc)  │
│         T.copy(gm, shared)│      │ → .o / .aibin            │
│         T.gemm(...)      │       └─────────────────────────┘
│     return kernel        │
└─────────────────────────┘
```

### 2.1.2 两种 JIT 模式

TileLang 支持两种编程模式：

| 模式 | 语法 | 特点 | 适用场景 |
|------|------|------|---------|
| **lazy 模式** | 函数显式 `return kernel`（PrimFunc） | 返回编译后的 kernel 对象，可多次调用 | 生产环境、性能优化 |
| **eager 模式** | DSL builder 模式（张量注解） | 立即编译并执行，返回结果 | 快速原型、调试 |

```python
# lazy 模式（推荐）
@tilelang.jit(target="ascend", out_idx=[2])
def matmul_kernel(M, N, K, dtype="bfloat16"):
    @T.prim_func
    def main(A: T.Buffer((M, K), dtype),
             B: T.Buffer((N, K), dtype),
             C: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, 256), T.ceildiv(N, 256)) as (bx, by):
            # ... tile 级计算
            pass
    return main

kernel = matmul_kernel(8192, 8192, 8192)
C = kernel(A, B)  # 可多次调用

# eager 模式（调试用）
result = tilelang.eager_add(A, B)  # 立即执行
```

### 2.1.3 后端支持

通过 `target` 参数切换后端：

| target | 编译器 | 硬件 | 专用 API |
|--------|-------|------|---------|
| `"cuda"` | nvcc | NVIDIA GPU | TMA / cp.async / warp specialized |
| `"ascend"` | bisheng (CCE) | 华为 Ascend NPU | SimdVF / SIMD MicroAPI / NZ 布局 |
| `"hip"` | hipcc | AMD ROCm | — |
| `"metal"` | metal compiler | Apple Metal | — |
| `"c"` | gcc | CPU | — |

### 2.1.4 编译流程概览

```
@tilelang.jit
  → tilelang.compile → tilelang.lower
    → resolve_pipeline(target)        # 选择 pass pipeline
      ├─ "ascend" → AscendPassPipelineBody
      └─ "cuda"   → CUDAPassPipelineBody
    → device_codegen                  # 代码生成
      ├─ ascend: bisheng → .aibin
      └─ cuda:   nvcc → .cubin
  → adapter (tvm_ffi / cython / ...)
  → JITKernel (可调用)
```

> 详细 pass 流水线 → 详见第 3 篇 § 3.1

### 2.1.5 核心装饰器

| 装饰器 | 作用 | 示例 |
|--------|------|------|
| `@T.prim_func` | 声明 PrimFunc（主计算函数） | `@T.prim_func` def main(...): |
| `@T.macro` | 声明宏（编译期展开，类似内联函数） | `@T.macro` def swizzle(idx): ... |
| `@tilelang.jit` | JIT 编译装饰器 | `@tilelang.jit(target="ascend")` |

### 2.1.6 Hello World — 最简 GEMM

```python
import tilelang
import tilelang.language as T

@tilelang.jit(target="ascend", out_idx=[2])
def hello_gemm(M=512, N=512, K=512, dtype="bfloat16"):
    TILE_M, TILE_N, TILE_K = 128, 128, 64
    NUM_BLOCKS = 32

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((N, K), dtype),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, TILE_M), T.ceildiv(N, TILE_N), threads=NUM_BLOCKS) as (bm, bn):
            A_L1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            B_L1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            C_L0C = T.alloc_l0c((TILE_M, TILE_N), "float32")

            T.copy(A[bm*TILE_M:(bm+1)*TILE_M, :K], A_L1)
            T.copy(B[bn*TILE_N:(bn+1)*TILE_N, :K], B_L1)
            T.gemm(A_L1, B_L1, C_L0C, transpose_B=True, clear_accum=True)
            T.copy(C_L0C, C[bm*TILE_M:(bm+1)*TILE_M, bn*TILE_N:(bn+1)*TILE_N])

    return main

# 编译并调用
kernel = hello_gemm()
C = kernel(A, B)
```

---

## 2.2 内存分配与层级

### 2.2.1 GPU 内存

| API | 层级 | 容量 | 说明 |
|-----|------|------|------|
| `T.alloc_shared(shape, dtype)` | shared memory | ~48KB/SM | SM 内所有线程共享 |
| `T.alloc_fragment(shape, dtype)` | register | — | 每线程私有寄存器 |
| `T.alloc_local(shape, dtype)` | local memory | — | 线程私有（可能 spill 到 GM） |

### 2.2.2 Ascend 内存层级（6 级，从远到近）

```
┌──────────────────────────────────────────────────────────────┐
│                     Ascend 内存层级                          │
│                                                              │
│  GM (HBM)  ──MTE2──→  UB  ──直接──→  Fragment (寄存器)      │
│   96 GB              256KB/core         2048 bit              │
│      │                 ↕                                    │
│      │               Vector 引擎                             │
│      │                 ↕                                    │
│      └──MTE2──→  L1  ──MTE1──→  L0A/L0B  ──Cube──→  L0C    │
│                   512KB/core              累加器(仅fp32)      │
│                         ↕                                    │
│                       Cube 引擎                              │
│                         ↕                                    │
│                      FixPipe → UB / GM                       │
└──────────────────────────────────────────────────────────────┘
```

| 层级 | API | 容量 | 引擎 | 用途 | 访问延迟 |
|------|-----|------|------|------|---------|
| **GM (HBM)** | `T.Buffer` / `T.Tensor` | 96 GB | — | 全局显存，所有核可见 | 最高 |
| **UB** | `T.alloc_shared(shape, dtype)` | 256 KB/core | MTE2(读) / MTE3(写) / VEC | 片上缓存，最常用 | 中 |
| **L1 (CBuf)** | `T.alloc_l1(shape, dtype)` | 512 KB/core | MTE2(GM→L1) / Cube | Cube 输入缓冲 | 中 |
| **L0A / L0B** | `T.alloc_l0a / T.alloc_l0b` | — | MTE1(L1→L0) | Cube 操作数寄存器 | 低 |
| **L0C** | `T.alloc_l0c(shape, "float32")` | — | Cube(累加) / FixPipe(搬出) | GEMM 累加器（仅 fp32） | 低 |
| **Fragment** | `T.alloc_fragment(shape, dtype)` | 2048 bit | 直接读写 | Vector 寄存器，零延迟 | 最低 |
| **Reducer** | `T.alloc_reducer(shape, dtype, op="sum")` | — | `T.finalize_reducer` | 跨线程归约专用 | 最低 |

### 2.2.3 关键决策：数据放哪一层

数据放置由**算子类型**决定：

| 算子类型 | 输入存放 | 计算位置 | 输出存放 |
|---------|---------|---------|---------|
| GEMM (T.gemm) | L1 → L0A/L0B | Cube → L0C | L0C → UB → GM |
| Vector (SimdVF) | GM → UB | Fragment | Fragment → UB → GM |
| Flash Attention | L1 (Q,K,V) + UB (softmax) | Cube + Vector | UB → GM |
| 归约 (RMSNorm) | GM → UB | Fragment + Reducer | Fragment → GM |

> 策略展开：数据放置的优化策略 → 详见第 5 篇 § 5.3

### 2.2.4 代码示例：各层级分配

```python
with T.Kernel(num_blocks) as pid:
    # GM → UB（最常见，Vector 算子用）
    shared = T.alloc_shared((BM, BK), dtype)        # UB, 256KB/core
    T.copy(gm_a[pid*BM:], shared)                   # MTE2: GM → UB

    # GM → L1（GEMM 用）
    a_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)      # L1, 512KB/core
    b_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
    T.copy(gm_a[...], a_l1)                         # MTE2: GM → L1

    # L0C（GEMM 累加器）
    c_l0c = T.alloc_l0c((TILE_M, TILE_N), "float32")

    # Fragment（Vector 寄存器）
    frag = T.alloc_fragment((64,), "float32")       # 2048 bit 寄存器

    # Reducer（跨线程归约）
    reducer = T.alloc_reducer((1,), "float32", op="sum")
    T.finalize_reducer(reducer, frag)               # 硬件级归约
```

---

## 2.3 循环与调度

### 2.3.1 T.Parallel — 并行循环

```python
def Parallel(
    *extents: int | PrimExpr,
    coalesced_width: int | None = None,
    loop_layout: Any | None = None,
    prefer_async: bool | None = None,
    annotations: dict | None = None,
) -> ForFrame
```

| 参数 | 说明 |
|------|------|
| `extents` | 各维度迭代范围 |
| `coalesced_width` | 合并访问宽度 |
| `loop_layout` | 并行循环布局提示（Fragment） |
| `prefer_async` | 请求 cp.async 注入（CUDA） |

```python
# 并行处理 BM 行
for i in T.Parallel(BM):
    out[i] = a[i] + b[i]
```

### 2.3.2 T.Persistent — 持久化调度

```python
def Persistent(
    domain: list[PrimExpr],       # tile 总数
    wave_size: PrimExpr,          # 核数（block 数）
    index: PrimExpr,              # tile 索引变量
    group_size: PrimExpr | int = 8,  # 工作窃取粒度
    num_stages: int = 0,          # 流水深度
    annotations: dict | None = None,
) -> ForFrame
```

| 参数 | 说明 | 典型值 |
|------|------|-------|
| `domain` | tile 总数列表 | `[M_TILES * N_TILES]` |
| `wave_size` | 核数（block 数） | `32`（Cube）或 `64`（Vector） |
| `index` | tile 索引变量 | `tile_idx` |
| `group_size` | 工作窃取粒度 | `1`（最细粒度）/ `8`（默认） |
| `num_stages` | 流水深度 | `0`（禁用）/ `2` / `3` / `4` |

```python
# 持久化调度：32 个核处理 OUT_TILES 个 tile
with T.Kernel(NUM_BLOCKS) as bx:
    for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
        m_tile = tile_idx // N_TILES
        n_tile = tile_idx % N_TILES
        # ... 处理 (m_tile, n_tile) tile
```

> **group_size 语义**：`group_size=1` 表示每核 1 tile，最细粒度负载均衡；`group_size=8` 表示每核连续处理 8 tiles，减少调度开销。
>
> 策略展开：group_size / num_stages 选择策略 → 详见第 5 篇 § 5.5

### 2.3.3 T.Pipelined — 软件流水

```python
def Pipelined(
    start: PrimExpr,
    stop: PrimExpr | None = None,
    num_stages: int = 0,           # 流水深度（buffer 份数）
    order: list[int] | None = None,  # 手动发射顺序
    stage: list[int] | None = None,  # 手动 stage 分配
    sync: list[list[int]] | None = None,  # 手动同步
    group: list[list[int]] | None = None,  # 手动分组
    annotations: dict | None = None,
) -> ForFrame
```

| 参数 | 说明 | 典型值 |
|------|------|-------|
| `start` | 迭代起点 | `0` |
| `stop` | 迭代终点 | `K_TILES` |
| `num_stages` | 流水深度（buffer 份数） | `2`（双缓冲）/ `3` / `4` |
| `order` | 手动发射顺序 | `[0, 1, 2]`（copy, gemm, store） |
| `stage` | 手动 stage 分配 | `[0, 1, 2]` |

```python
# K 维流水：num_stages=2 双缓冲
for kt in T.Pipelined(K_TILES, num_stages=2):
    T.copy(A[..., kt*TILE_K:], A_L1)       # stage 0: MTE2 加载
    T.copy(B[..., kt*TILE_K:], B_L1)       # stage 0: MTE2 加载
    T.gemm(A_L1, B_L1, C_L0C, clear_accum=(kt==0))  # stage 1: Cube 计算

# 手动流水（高级）
for kt in T.Pipelined(K_TILES, num_stages=3, order=[0,1,2], stage=[0,1,2]):
    T.copy(A[..., kt*TILE_K:], A_L1)       # order=0, stage=0
    T.gemm(A_L1, B_L1, C_L0C)              # order=1, stage=1
    T.copy(C_L0C, C[..., kt*TILE_K:])      # order=2, stage=2
```

> **num_stages 语义**：N 级流水有 N 份 buffer 在途。`num_stages=2` 时，MTE2 加载 frame k+1 的同时，Cube 计算 frame k。
>
> 策略展开：num_stages 选择策略 → 详见第 5 篇 § 5.5.1

### 2.3.4 T.serial — 串行循环

```python
for i in T.serial(N):
    # 串行执行，无并行/流水
    pass
```

### 2.3.5 三种循环对比

```python
# ① T.Parallel: 并行执行
for i in T.Parallel(BM):
    out[i] = a[i] + b[i]       # 所有 i 并行

# ② T.Persistent: 持久化 + 工作窃取
for tile in T.Persistent([N], num_cores, pid):
    process(tile)              # 核间分配 tile，grid-stride

# ③ T.Pipelined: 软件流水
for k in T.Pipelined(K, num_stages=3):
    load(k)                    # 3 级流水：load/compute/store 重叠
    compute(k)
    store(k)
```

### 2.3.6 PersistentTileScheduler

CUTLASS 风格 swizzle + cluster 调度（CUDA Hopper 专用）：

```python
# swizzle 重排 tile 访问顺序，提升 L2 局部性
with T.Kernel(NUM_BLOCKS, thread_entity=PersistentTileScheduler) as bx:
    # ...
```

---

## 2.4 数据搬运 T.copy

### 2.4.1 完整签名

```python
def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    prefer_instruction: str | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
    transpose: bool = False,              # Ascend: GM→L1 dn2nz
    l2_cache_ctrl: int | str | None = None,  # Ascend: L2 缓存策略
    unit_flag_ctrl: int | PrimExpr | None = None,  # Ascend: FixPipe 重叠
    sub_blockid: int | PrimExpr | None = None,  # Ascend: AIV 子核选择
    scale: BufferLikeType | None = None,  # Ascend: MX scale-factor
    pad_value: int | float | PrimExpr | None = None,  # Ascend: 填充值
    data_select: bool = False,            # Ascend: 复用已设 pad 值
) -> PrimExpr | Stmt
```

### 2.4.2 参数详解

| 参数 | 类型 | 说明 | 适用后端 |
|------|------|------|---------|
| `src` / `dst` | Buffer / BufferRegion / BufferLoad | 源/目标内存区域 | 所有 |
| `coalesced_width` | int | 合并访问宽度 | CUDA |
| `disable_tma` | bool | 禁用 TMA 加速 | CUDA |
| `eviction_policy` | str | cache 驱逐策略 | CUDA |
| `prefer_instruction` | str | 首选 lowering 指令 ("tma"/"cp_async") | CUDA |
| `transpose` | bool | GM→L1 转置提示（dn2nz 布局转换） | Ascend |
| `l2_cache_ctrl` | int/str | L2 缓存控制策略 | Ascend |
| `unit_flag_ctrl` | int/Expr | FixPipe 重叠控制 | Ascend |
| `sub_blockid` | int/Expr | AIV 子核选择器 | Ascend |
| `scale` | Buffer | MX scale-factor 源 buffer（L1→L0A/L0B） | Ascend |
| `pad_value` | int/float/Expr | GM→UB 填充值（32B 对齐） | Ascend |
| `data_select` | bool | 复用已设 pad 值（与 pad_value 互斥） | Ascend |

### 2.4.3 l2_cache_ctrl 选项

| 字符串 | 整数值 | 说明 | 适用路径 |
|--------|-------|------|---------|
| `"NORMAL_FV"` | 0 | 正常 L2 分配（默认 load） | GM→UB / GM→L1 |
| `"NORMAL_LV"` | 1 | 正常 L2，last victim | GM→UB / GM→L1 |
| `"NOTALLOC_KEEP"` | 4 | 旁路 L2，流式访问（默认 store） | GM→UB / GM→L1 / UB→GM |
| `"NOTALLOC_CLEAN"` | 5 | 旁路 L2 + 清除 L2 行 | GM→UB / GM→L1 |
| `"NOTALLOC_DROP"` | 6 | 旁路 L2 + 丢弃 | GM→UB / GM→L1 |
| `"NORMAL_RED"` | 3 | 正常 L2，reduction store | UB→GM |
| `"NOTALLOC_PW"` | 5 | 部分写策略 | UB→GM |
| `"WBH_FV"` | 8 | Write-back hint, first victim | UB→GM |
| `"WTS_FV"` | 12 | Write-through, first victim | UB→GM |

> 策略选择决策树 → 详见第 5 篇 § 5.3.1

### 2.4.4 三种典型搬运

```python
# ① GM → UB（最常见，Vector 算子用）
shared = T.alloc_shared((BM, BK), dtype)
T.copy(gm_a[pid*BM:(pid+1)*BM, :BK], shared)

# ② GM → L1（GEMM 用，需 transpose=True 做 dn2nz）
a_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
T.copy(gm_a[m_start:m_end, k_start:k_end], a_l1, transpose=True)

# ③ L0C → GM（GEMM 输出，需 unit_flag_ctrl 做 FixPipe 重叠）
T.copy(c_l0c, gm_c[m_start:m_end, n_start:n_end], unit_flag_ctrl=UF_3)
```

### 2.4.5 T.dual_copy — L0C 跨子核拆分

```python
# L0C 累加结果跨 2 个 AIV 子核拆分输出
temp = T.alloc_shared((TILE_M // 2, TILE_N), out_dtype)
T.dual_copy(c_l0c, temp, unit_flag_ctrl=UF_3)       # L0C → UB (M-split)
T.dual_copy(temp, gm_c[...])                         # UB → GM
```

**M-split 语义**：L0C 的 `TILE_M` 行拆分到 2 个 AIV 子核，每个子核处理 `TILE_M/2` 行。

### 2.4.6 pad_value — 非对齐填充

```python
# 非对齐尾块填充到 32B 边界
T.copy(gm_a[pid*BM:(pid+1)*BM, :D], shared, pad_value=0)
# 若 D 不是 32B 对齐，尾部填充 0 到下一个 32B 边界
```

---

## 2.5 T.gemm 与 Cube 引擎

### 2.5.1 签名

```python
T.gemm(
    a: Buffer,           # L0A 输入，shape (M, K)
    b: Buffer,           # L0B 输入，shape (N, K)
    c: Buffer,           # L0C 输出，shape (M, N), dtype=float32
    transpose_B: bool = True,   # 强制 True，权重必须 [N, K] 布局
    clear_accum: bool | PrimExpr = False,  # 首帧清零
    unit_flag_ctrl: int | PrimExpr | None = None,  # FixPipe 重叠
    # ... 其他参数
)
```

### 2.5.2 关键约束

| 约束 | 说明 |
|------|------|
| `transpose_B=True` | **强制**，权重必须 `[N, K]` 布局（非 `[K, N]`） |
| `c.dtype = "float32"` | L0C 累加器仅支持 float32 |
| `a` 在 L0A / L1 | 输入需在 L0A 或 L1（自动 L1→L0A 搬运） |
| `b` 在 L0B / L1 | 输入需在 L0B 或 L1（自动 L1→L0B 搬运） |

### 2.5.3 完整 GEMM 示例

```python
@T.prim_func
def main(
    X: T.Buffer((M, K), dtype),
    W: T.Buffer((N, K), dtype),   # 注意：W 是 [N, K] 布局
    C: T.Buffer((M, N), out_dtype),
):
    with T.Kernel(NUM_BLOCKS) as bx:
        res = T.alloc_l0c((TILE_M, TILE_N), "float32")
        x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
        w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)

        for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
            m_tile, n_tile = aswt_swizzle(tile_idx)
            for kt in T.Pipelined(K_TILES, num_stages=2):
                # GM → L1（dn2nz 转置）
                T.copy(X[m_tile*TILE_M:(m_tile+1)*TILE_M, kt*TILE_K:(kt+1)*TILE_K], x_l1)
                T.copy(W[n_tile*TILE_N:(n_tile+1)*TILE_N, kt*TILE_K:(kt+1)*TILE_K], w_l1)
                # L1 × L1 → L0C（Cube GEMM）
                T.gemm(x_l1, w_l1, res,
                       transpose_B=True,
                       clear_accum=(kt == 0),
                       unit_flag_ctrl=T.Select(kt == K_TILES-1, UF_3, UF_2))
            # L0C → GM（FixPipe 输出）
            T.copy(res, C[m_tile*TILE_M:(m_tile+1)*TILE_M, n_tile*TILE_N:(n_tile+1)*TILE_N],
                  unit_flag_ctrl=UF_3)
```

### 2.5.4 clear_accum 语义

```python
T.gemm(a, b, c, clear_accum=(kt == 0))
# kt=0 时：c = a × b^T  （清零后累加）
# kt>0 时：c += a × b^T  （累加）
```

### 2.5.5 unit_flag_ctrl 语义

| 值 | 含义 | 适用帧 |
|----|------|-------|
| `0` (UF_NONE) | 不重叠 | — |
| `2` (UF_2) | 中间帧 FixPipe 重叠 | `kt = 0..K_TILES-2` |
| `3` (UF_3) | 尾帧 FixPipe 重叠 | `kt = K_TILES-1` |

> 策略展开：FixPipe 重叠策略 → 详见第 5 篇 § 5.4.2

### 2.5.6 HF32 高精度模式

```python
T.set_hf32_mode("nearest_zero")  # 或 "nearest_even"
# 启用后 Cube 在 fp32 模式下使用 HF32（高精度 float32）
```

---

## 2.6 SimdVF / SimtVF — 两种向量执行模型

### 2.6.1 SimdVF — 向量 SIMD 执行模型

```python
def SimdVF(latency: int = 0) -> SimdVFFrame
```

| 参数 | 说明 |
|------|------|
| `latency` | 测量的 SimdVF 延迟（cycles），注入 `tl.vf_latency` 标记供 Z3 auto-schedule |

**特点**：
- **无线程概念**，纯向量寄存器级 SIMD
- 单核一次处理一个 2048-bit 向量寄存器
- 使用 `T.simd.*` MicroAPI 直接操作 SIMD 寄存器
- 适合：规则向量计算、归约、类型转换

```python
with T.SimdVF(latency=120):
    # 在此区域内使用 SIMD MicroAPI
    vreg = T.alloc_fragment((128,), "bfloat16")
    S.vld(vreg, ub_buffer[0])          # 加载
    S.vcmax(vreg, vreg)                # 跨 lane 归约
    S.vsts(ub_buffer[0], vreg)         # 存储
```

### 2.6.2 SimtVF — SIMT 线程并行执行模型

```python
def SimtVF(
    threads: int | list[int] | tuple = 128,  # 线程数
    latency: int = 0,
) -> SimtVFFrame
```

| 参数 | 说明 |
|------|------|
| `threads` | 线程数，可传 int（仅 x）或 list（x, y, z） |
| `latency` | 测量的 SimtVF 延迟（cycles） |

**特点**：
- **SIMT 线程并行**，类似 GPU 线程模型
- 每线程处理 1 元素，`T.Parallel` 并行
- 自动插入 `asc_syncthreads` 同步
- 适合：不规则 gather、逐元素操作、需要线程级分支

```python
with T.SimtVF(threads=128):
    # 128 个线程并行
    for i in T.Parallel(128):
        out[i] = a[i] + b[i]
```

### 2.6.3 向量寄存器宽度

2048-bit 向量寄存器，按 dtype 计算 lanes：

| dtype | elem bits | lanes (2048/bits) | 说明 |
|-------|----------|-------------------|------|
| float32 | 32 | 64 | 一次处理 64 个 fp32 |
| bfloat16 | 16 | 128 | 一次处理 128 个 bf16 |
| float16 | 16 | 128 | 一次处理 128 个 fp16 |
| int8 | 8 | 256 | 一次处理 256 个 int8 |
| float8_e4m3 | 8 | 256 | 一次处理 256 个 fp8 |
| float4_e2m1 | 4 | 512 | 一次处理 512 个 fp4 |

### 2.6.4 选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 归约（amax/sum） | SimdVF | `vcmax`/`vcadd` 跨 lane 归约高效 |
| 元素级（cast/scale） | SimdVF | 连续向量，无分支 |
| 类型转换（vcvt） | SimdVF | SIMD 指令直接转换 |
| online softmax | SimdVF | `vexpdif` 融合指令 |
| Gather/Scatter | SimtVF | 不规则访存，需线程级分支 |
| 条件逻辑 | SimtVF | 需线程级 if-else |
| TopK 选择 | SimdVF | `vcmp` + `vsel` 向量比较选择 |

> 策略展开：SimdVF vs SimtVF 深度选择策略 → 详见第 5 篇 § 5.2.4

### 2.6.5 同一算子两种执行模型对比

```python
# SimdVF 版本（推荐，性能更优）
with T.SimdVF():
    vreg = T.alloc_fragment((128,), "bfloat16")
    S.vld(vreg, ub_in[0])
    S.vmax(vreg, vreg, vreg)    # 向量 max
    S.vsts(ub_out[0], vreg)

# SimtVF 版本（更灵活，但性能可能稍差）
with T.SimtVF(threads=128):
    for i in T.Parallel(128):
        out[i] = max(a[i], b[i])
```

---

## 2.7 SIMD MicroAPI 速查

> 导入方式：`import tilelang.language.simd as S` 或 `from tilelang.ascend.language.simd import *`

### 2.7.1 加载/存储

| API | 功能 | 关键参数 |
|-----|------|---------|
| `S.vld(dst, src)` | 向量加载 | — |
| `S.vld2(dst1, dst2, src)` | 交错加载（一次加载 256 bf16 → 两个 128-lane） | — |
| `S.vsts(dst, src)` | 向量存储 | — |
| `S.vsstb(dst, src, stride=128)` | NZ 布局存储（stride=128 for NZ packing） | `stride` |

### 2.7.2 算术

| API | 功能 | 说明 |
|-----|------|------|
| `S.vadd(dst, a, b)` | 逐元素加 | — |
| `S.vsub(dst, a, b)` | 逐元素减 | — |
| `S.vmul(dst, a, b)` | 逐元素乘 | — |
| `S.vdiv(dst, a, b)` | 逐元素除 | — |
| `S.vmax(dst, a, b)` | 逐元素 max | — |
| `S.vmin(dst, a, b)` | 逐元素 min | — |
| `S.vand(dst, a, b)` | 位与 | 清符号位: `vand(x, 0x7FFF)` 替代 `vabs` |
| `S.vor(dst, a, b)` | 位或 | — |
| `S.vxor(dst, a, b)` | 位异或 | — |
| `S.vshl(dst, a, shift)` | 左移 | — |
| `S.vshr(dst, a, shift)` | 右移 | — |

### 2.7.3 FMA（融合乘加）

| API | 功能 | 说明 |
|-----|------|------|
| `S.vmula(dst, a, b)` | `dst = dst + a * b` | 融合乘加，减少寄存器往返 |
| `S.vmadd(dst, a, b, c)` | `dst = a * b + c` | — |
| `S.vaxpy(dst, a, x, y)` | `dst = a * x + y` | — |

### 2.7.4 归约

| API | 功能 | 说明 |
|-----|------|------|
| `S.vcadd(dst, src)` | 跨 lane 求和 | `dst[0] = sum(src[i])` |
| `S.vcmax(dst, src)` | 跨 lane 求最大 | `dst[0] = max(src[i])` |
| `S.vcmin(dst, src)` | 跨 lane 求最小 | `dst[0] = min(src[i])` |
| `S.vcgadd(dst, src)` | 跨 lane 群求和 | — |
| `S.vcgmax(dst, src)` | 跨 lane 群求最大 | bf16 归约高效 |
| `S.vcgmin(dst, src)` | 跨 lane 群求最小 | — |

### 2.7.5 交错/去交错

| API | 功能 | 说明 |
|-----|------|------|
| `S.vintlv(dst1, dst2, a, b)` | 交错合并 | — |
| `S.vdintlv(dst1, dst2, src)` | 去交错拆分 | — |

### 2.7.6 类型转换

```python
S.vcvt(dst, src, target_dtype="float8_e4m3fn", part=0, round="nearest_zero", sat=True)
```

| 参数 | 说明 |
|------|------|
| `target_dtype` | 目标类型 |
| `part` | 部分（fp8 有 part0/part1） |
| `round` | 舍入模式：`"nearest_zero"` / `"nearest_even"` / `"stochastic"` |
| `sat` | 是否饱和 |

### 2.7.7 Gather/Scatter

| API | 功能 |
|-----|------|
| `S.vgatherb(dst, indices, src)` | 按 index gather |
| `S.vgather2(dst, indices, src)` | 2-element gather |
| `S.vscatter(dst, indices, src)` | 按 index scatter |

### 2.7.8 排列/选择

| API | 功能 | 说明 |
|-----|------|------|
| `S.vci(dst, src)` | 向量压缩 | — |
| `S.vcmp(dst, a, b)` | 向量比较 | 产生谓词 mask |
| `S.vsel(dst, mask, a, b)` | 条件选择 | `dst = mask ? a : b` |
| `S.vselr(dst, mask, src)` | 条件选择保留 | — |
| `S.vpack(dst, a, b)` | 打包 | — |

### 2.7.9 谓词

| API | 功能 |
|-----|------|
| `S.pset(elem_width, dist)` | 设置谓词 mask |
| `S.pge(elem_width, dist)` | 大于等于谓词 |
| `S.pand(dst, a, b)` | 谓词与 |
| `S.por(dst, a, b)` | 谓词或 |
| `S.pxor(dst, a, b)` | 谓词异或 |
| `S.pnot(dst, src)` | 谓词非 |
| `S.psel(dst, mask, a, b)` | 谓词选择 |

### 2.7.10 其他

| API | 功能 |
|-----|------|
| `S.vdupv(dst, src)` | 广播标量到所有 lane |
| `S.vexpdif(dst, src, max)` | `exp(src - max)` 融合（softmax 关键） |
| `S.mem_bar("VST_VLD")` | 存储→加载屏障 |
| `S.pset(32, "PAT_ALL")` | 全 lane 模式 |

> 与第 1 篇呼应：`vld2`/`vcgmax` 对应 per_token_cast 优化案例；`vand` 对应"寄存器复用"法则

---

## 2.8 PassConfig 详解

> 导入：`from tilelang.transform import PassConfigKey`

### 2.8.1 TL_ 系列（TileLang 专属）

| PassConfigKey | 说明 | 典型值 |
|---------------|------|-------|
| `TL_ENABLE_FAST_MATH` | 快速数学（忽略 NaN/Inf） | `True` |
| `TL_DISABLE_DATA_RACE_CHECK` | 禁用数据竞争检查 | `True`（NPU 常用） |
| `TL_PTXAS_REGISTER_USAGE_LEVEL` | ptxas 寄存器使用级别 | `0-10` |
| `TL_ENABLE_AUTO_SCHEDULE` | 启用 Z3 auto-schedule | `True` |
| `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` | 激进 shared memory 合并 | `True` |
| `TL_DISABLE_SHARED_MEMORY_REUSE` | 禁用 shared memory 复用 | `False` |
| `TL_ENABLE_DUMP_IR` | dump 每轮 IR | `True`（调试） |
| `TL_DUMP_IR_DIR` | dump 目录 | `"/tmp/ir_dump"` |
| `TL_AST_PRINT_ENABLE` | 打印 AST | `True`（调试） |
| `TL_LAYOUT_VISUALIZATION_ENABLE` | 布局可视化 | `True`（调试） |
| `TL_PASS_PROFILE` | pass 耗时分析 | `True`（调试） |
| `TL_PASS_PROFILE_THRESHOLD_MS` | pass 耗时阈值 | `100` |

### 2.8.2 TIR_ 系列（TIR 相关）

| PassConfigKey | 说明 | 典型值 |
|---------------|------|-------|
| `TIR_DISABLE_VECTORIZE` | 禁用向量化 lower | `True`（NPU 常用） |
| `TIR_DISABLE_STORAGE_REWRITE` | 禁用存储重写 | `False` |

### 2.8.3 NPU 常用组合

```python
from tilelang.transform import PassConfigKey

@tilelang.jit(
    target="ascend",
    pass_configs={
        PassConfigKey.TL_ENABLE_FAST_MATH: True,
        PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: True,
    },
)
def kernel(...):
    ...
```

> 注：这些 config 如何影响编译 pass → 详见第 3 篇 § 3.2-3.3；调优实践 → 详见第 5 篇 § 5.6

---

## 2.9 JIT 编译与调试

### 2.9.1 @tilelang.jit 参数

```python
@tilelang.jit(
    target="ascend",              # 目标后端
    out_idx=[2],                  # 输出 buffer 索引
    pass_configs={...},           # PassConfig
    execution_backend="auto",     # 执行后端
    verbose=False,                # 详细输出
    debug_root_path=None,         # 调试文件根路径
    compile_flags=["--cce-res-usage"],  # 编译器选项
)
```

### 2.9.2 execution_backend 选项

| 值 | 说明 | 适用场景 |
|----|------|---------|
| `"auto"` | 自动选择 | 默认 |
| `"dlpack"` | DLPack | 跨框架 |
| `"tvm_ffi"` | TVM runtime | 默认（Ascend） |
| `"cython"` | Cython | 性能优化 |
| `"nvrtc"` | NVRTC | CUDA 运行时编译 |
| `"torch"` | PyTorch | PyTorch 集成 |
| `"cutedsl"` | CuTe DSL | CUDA DSL |

### 2.9.3 调试技巧

```python
# ① dump 每轮 IR
@tilelang.jit(
    target="ascend",
    pass_configs={
        PassConfigKey.TL_ENABLE_DUMP_IR: True,
        PassConfigKey.TL_DUMP_IR_DIR: "/tmp/ir_dump",
    },
)
def kernel(...):
    ...

# ② 打印 AST
PassConfigKey.TL_AST_PRINT_ENABLE: True

# ③ 布局可视化
PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True

# ④ pass 耗时分析
PassConfigKey.TL_PASS_PROFILE: True
PassConfigKey.TL_PASS_PROFILE_THRESHOLD_MS: 100  # 超过 100ms 的 pass
```

### 2.9.4 编译缓存

TileLang JIT 自动缓存编译结果。清除缓存：

```python
# 清除 JIT 缓存
tilelang.clear_cache()

# 或手动删除缓存目录
import shutil
shutil.rmtree("~/.tilelang/cache", ignore_errors=True)
```

### 2.9.5 完整调试示例

```python
import tilelang
import tilelang.language as T
from tilelang.transform import PassConfigKey

@tilelang.jit(
    target="ascend",
    out_idx=[2],
    pass_configs={
        PassConfigKey.TL_ENABLE_FAST_MATH: True,
        PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        PassConfigKey.TL_ENABLE_DUMP_IR: True,       # 调试
        PassConfigKey.TL_DUMP_IR_DIR: "/tmp/gemm_ir", # IR dump 路径
    },
    verbose=True,
)
def debug_gemm(M=512, N=512, K=512):
    @T.prim_func
    def main(A, B, C):
        with T.Kernel(...) as (bx, by):
            # ...
            pass
    return main

kernel = debug_gemm()
# 查看 /tmp/gemm_ir/ 下的每轮 IR
```

---

## 2.10 读者检查点

1. **Ascend 有几级内存？UB 和 L1 的容量和用途分别是什么？**
   - 6 级：GM(96GB) → UB(256KB/core, Vector) → L1(512KB/core, Cube) → L0A/L0B → L0C → Fragment(2048bit)
   - UB 用于 Vector 算子片上缓存；L1 用于 GEMM Cube 输入缓冲

2. **T.copy 的 l2_cache_ctrl 有哪些选项？各自适用什么场景？**
   - `NORMAL_FV`(默认 load): 数据复用场景
   - `NOTALLOC_KEEP`(默认 store): 一次性大数据流式访问
   - `NOTALLOC_CLEAN`: 旁路 + 避免污染
   - （策略选择 → 详见第 5 篇 § 5.3.1）

3. **SimdVF 和 SimtVF 的区别是什么？2048-bit 寄存器对 bf16 有多少 lanes？**
   - SimdVF: 无线程，纯向量寄存器 SIMD；SimtVF: SIMT 线程并行
   - bf16: 2048/16 = 128 lanes

4. **写一个最简 GEMM kernel 骨架（T.copy + T.gemm + T.copy）**

```python
with T.Kernel(num_blocks) as bx:
    a_l1 = T.alloc_l1((M, K), dtype)
    b_l1 = T.alloc_l1((N, K), dtype)
    c_l0c = T.alloc_l0c((M, N), "float32")
    T.copy(gm_a, a_l1)
    T.copy(gm_b, b_l1)
    T.gemm(a_l1, b_l1, c_l0c, transpose_B=True, clear_accum=True)
    T.copy(c_l0c, gm_c)
```

5. **T.Persistent 的 group_size 和 num_stages 分别控制什么？**
   - `group_size`: 工作窃取粒度（1=最细粒度，8=默认）
   - `num_stages`: 流水深度（buffer 份数）

6. **NPU 常用的 PassConfig 组合是什么？**
   - `TL_ENABLE_FAST_MATH=True` + `TIR_DISABLE_VECTORIZE=True` + `TL_DISABLE_DATA_RACE_CHECK=True`

---

## 2.11 小结

本篇系统介绍了 TileLang NPU 编程模型的全部 API：

| 类别 | 核心 API | 数量 |
|------|---------|------|
| 内存分配 | `alloc_shared` / `alloc_l1` / `alloc_l0c` / `alloc_fragment` / `alloc_reducer` | 7 |
| 循环调度 | `T.Parallel` / `T.Persistent` / `T.Pipelined` / `T.serial` | 4 |
| 数据搬运 | `T.copy` / `T.dual_copy` / `T.fill` | 3 |
| 计算 | `T.gemm` / `T.reduce_*` | 5+ |
| 执行模型 | `T.SimdVF` / `T.SimtVF` | 2 |
| SIMD MicroAPI | `vld` / `vsts` / `vadd` / `vcvt` / `vcmax` / ... | 50+ |
| 编译 | `@tilelang.jit` / `PassConfigKey` | 40+ |

**本篇讲了 API 的 What 和 How；策略选择（Why/When）→ 详见第 5 篇**

下一篇将深入编译后端，理解这些 API 如何 lowered 到硬件指令——这是从"会用"到"精通"的关键一步。
