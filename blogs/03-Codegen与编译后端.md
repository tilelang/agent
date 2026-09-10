# 第 3 篇：TileLang 框架 Codegen 与编译后端

> **定位**：编译原理深度解析。读完本篇，读者理解 API → 硬件指令的完整链路。
>
> **知识边界**：
> - ✅ 本篇讲：编译流程、pass pipeline 结构、pass 原理、Z3 建模、后端对比、调试方法
> - ❌ 本篇不讲：API 签名（→ 第 2 篇）、Pass 调优实践（→ 第 5 篇 § 5.6）、模板代码（→ 第 4 篇）
>
> **前置**：第 2 篇（了解 API 表面）　**后续**：第 5 篇利用编译 pass 做极致优化

---

## 3.1 编译流程全景

### 3.1.1 从 @tilelang.jit 到 kernel 加载

完整的编译链路从 `@tilelang.jit` 装饰器开始，经过 IR 变换、代码生成、适配器包装，最终返回可调用的 kernel 对象：

```
@tilelang.jit
  → JITImpl.__call__ → JITImpl.compile
    → tilelang.compile → cached → JITKernel
      → _compile_and_create_adapter
        → tilelang.lower (engine/lower.py)
          → lower_to_host_device_ir
            → PreLowerSemanticCheck          # 语义检查
            → resolve_pipeline(target)       # 选择 pass pipeline
                ├─ "ascend" → AscendPassPipelineBody
                ├─ "cuda"   → CUDAPassPipelineBody
                ├─ "hip"    → ROCMPassPipelineBody
                └─ "metal"  → MetalPassPipelineBody
            → Filter(host/device)            # 分离 host/device IR
          → device_codegen                   # 代码生成
              → resolve_device_codegen(target)
                ├─ cuda:    nvcc.compile_cuda    → .cubin
                ├─ cutedsl: CuTe DSL             → .cu
                ├─ ascend:  bisheng.compile_ascend → .aibin
                ├─ pto:     bisheng (PTO)        → .aibin
                ├─ hip:     hipcc                → .co
                └─ metal:   metal compiler      → .metallib
        → adapter                            # 运行时适配器
            ├─ TVMFFIKernelAdapter (默认)
            ├─ CythonKernelAdapter
            ├─ NVRTCKernelAdapter
            ├─ MetalKernelAdapter
            └─ CuTeDSLKernelAdapter
  → JITKernel (可调用对象)
```

### 3.1.2 两种 JIT 模式

```python
# lazy 模式：函数显式 return kernel（PrimFunc）
@tilelang.jit(target="ascend")
def my_kernel(M, N, K):
    @T.prim_func
    def main(A: T.Buffer(...), B: T.Buffer(...), C: T.Buffer(...)):
        # ... tile 级计算
        pass
    return main  # 返回 PrimFunc → 编译 → JITKernel

kernel = my_kernel(8192, 8192, 8192)  # 编译
C = kernel(A, B)                        # 可多次调用

# eager 模式：DSL builder 模式
result = tilelang.eager_op(A, B)  # 立即编译并执行
```

**自动推断**：`_infer_jit_mode` 检测函数体结构——有 `return prim_func` 则为 lazy，否则为 eager。

### 3.1.3 Kernel Adapter 层次

Adapter 层负责将编译后的 kernel 包装为 Python 可调用对象：

```
BaseKernelAdapter (抽象基类)
  ├── _convert_torch_func()    # torch.Tensor → TVM Buffer 转换
  ├── get_device_stream()      # 设备/流获取
  │
  ├── TVMFFIKernelAdapter      # TVM runtime + DLPack（默认）
  ├── CythonKernelAdapter      # Cython 包装
  ├── NVRTCKernelAdapter       # NVRTC 运行时编译
  ├── MetalKernelAdapter       # Metal 后端
  └── CuTeDSLKernelAdapter     # CuTe DSL 后端
```

---

## 3.2 通用 IR Pass 流水线

> 所有后端共享的 target 无关 pass。

### 3.2.1 前端规范化

| Pass | 作用 | 示例 |
|------|------|------|
| **BindTarget** | 绑定 target 信息到 IR | 附加 target 属性 |
| **MaterializeKernelLaunch** | 将 T.Kernel 的 thread_binding 展开为 thread_extent | `T.Kernel(N) as bx` → `blockIdx.x` |
| **LetInline** | let 表达式内联 | `let x = 1; x + x` → `1 + 1` |
| **LegalizeNegativeIndex** | 负索引合法化 | `A[-1]` → `A[N-1]` |
| **InjectAssumes** | 注入 assume 提示加速 TVM prover | `T.assume(cond)` → 优化约束 |
| **HoistBroadcastValues** | 广播值外提 | 减少重复计算 |
| **Simplify** | TileLang 增强化简 | 传递不等式证明、布尔 AND-of-ORs |
| **DecoupleTypeCast** | 解耦混合精度类型转换 | 插入中间 cast buffer |

### 3.2.2 布局推断与 Tile Op Lowering

| Pass | 作用 | 说明 |
|------|------|------|
| **LayoutReducer** | 归约器布局设置 | 为 `alloc_reducer` 设置布局 |
| **LayoutInference** | fragment/shared memory 布局推断 | 推断最优内存布局 |
| **PipelinePlanning** | 软件流水线规划 | 分析数据依赖，规划流水 |
| **InjectSoftwarePipeline** | 注入软件流水线代码 | 生成多缓冲 + 同步代码 |
| **LowerTileOp** | 高层 tile op → 低层 op | `T.copy` → DMA 指令，`T.gemm` → Cube 指令 |
| **ClusterPlanning** | cluster 规划（Hopper） | 多 SM cluster 协作 |

### 3.2.3 循环与内存优化

| Pass | 作用 | 说明 |
|------|------|------|
| **LoopUnswitching** | 循环外提不变 if | `for: if(c): ...` → `if(c): for: ...` |
| **LegalizeVectorizedLoop** | 向量化循环合法化 | 确保向量化合法 |
| **UnrollLoop** | 循环展开 | 小循环完全展开 |
| **VectorizeLoop** | 循环向量化 | 标量循环 → 向量指令 |
| **StorageRewrite** | 临时 buffer 复用 | inplace 检测 + 生命周期分析 |
| **FlattenBuffer** | buffer 扁平化 | 多维 buffer → 1D + offset |
| **MergeSharedMemoryAllocations** | shared memory 合并 | 合并多个 alloc 为一个连续分配 |

### 3.2.4 后端 lowering

| Pass | 作用 |
|------|------|
| **SplitHostDevice** | 分离 host/device 代码 |
| **MakePackedAPI** | 生成 packed API 接口 |
| **LowerDeviceKernelLaunch** | lowering kernel launch 代码 |
| **ThreadSync** | 插入线程同步（shared/global） |

---

## 3.3 Ascend 编译后端

### 3.3.1 Ascend Pass Pipeline 完整序列

以下是 `tilelang/ascend/pipeline.py` 中 `AscendPassPipelineBody` 的完整 pass 序列，分为三段：

```python
# ════════════════════════════════════════════════════════════════
# 前段：规范化 + 布局推断
# ════════════════════════════════════════════════════════════════
mod = BindTarget(target)(mod)
mod = MaterializeKernelLaunch()(mod)
mod = LetInline()(mod)                          # if should_force_let_inline
mod = AddWrapperForSingleBufStore()(mod)
mod = LegalizeNegativeIndex()(mod)
mod = VerifyParallelLoop()(mod)                 # if should_enable_race_check
mod = InjectAssumes()(mod)
mod = LegalizeSimdMerging()(mod)                # @Ascend: SIMD merging 合法化
mod = Simplify()(mod)
mod = UnrollLoopSkipVF()(mod)                   # @Ascend: VF 外 unroll
mod = Simplify()(mod)
mod = LayoutReducer()(mod)
mod = LayoutInference()(mod)
mod = InsertNd2Nz()(mod)                        # @Ascend: ND→NZ 布局转换
mod = VFChecker()(mod)                          # @Ascend: VF 合法性检查
mod = AscendInsertOOBPadding()(mod)             # @Ascend: OOB padding

# ════════════════════════════════════════════════════════════════
# Auto-Schedule 段：Z3 自动调度（核心）
# ════════════════════════════════════════════════════════════════
if allow_autoschedule(pass_ctx):
    mod = NormalizeControlFlowForSchedule()(mod)
    mod = AnnotateMultiBufferEligible()(mod)    # 标记可多缓冲的 buffer
    mod = NormalizeNoConflictHints()(mod)       # 处理 assume_no_conflict
    mod = NormalizeTask()(mod)                  # 任务归一化
    mod = LatencyEstimator()(mod)               # 估计各任务 latency
    mod = AutoSchedule()(mod)                   # ★ Z3 自动调度
    mod = WarpgroupPartition()(mod)             # warpgroup 划分
    mod = RestoreWhileLoops()(mod)              # 恢复 while 循环
    mod = NormalizeMixedKernelSid()(mod)        # MixedKernel SID 归一化
    mod = RewriteDualCopy()(mod)                # dual_copy 重写
    mod = Simplify()(mod)

# ════════════════════════════════════════════════════════════════
# 后段：lowering + codegen 准备
# ════════════════════════════════════════════════════════════════
mod = NormalizeBufferVersion()(mod)             # buffer 版本归一化
mod = AscendSimdVFLowerParallel()(mod)          # @Ascend: SIMD 向量 lowering
mod = LowerTileOp()(mod)                        # tile op → 低层 op
mod = DecoupleTypeCast()(mod)
mod = LegalizeVectorizedLoop()(mod)
mod = LegalizeSafeMemoryAccess()(mod)
mod = LowerAccessPtr()(mod)
mod = Simplify()(mod)
mod = HoistNonRestrictParams()(mod)
mod = HoistGlobalBufferAllocations()(mod)
mod = LowerOpaqueBlock()(mod)
mod = Simplify()(mod)
mod = RewriteAscendBufferVersionLayout()(mod)   # @Ascend: 多缓冲 32B 对齐
mod = NarrowDataType(32)(mod)
mod = FlattenBuffer()(mod)
mod = ConfigIndexBitwidth()(mod)
mod = Simplify()(mod)
mod = VectorizeLoop()(mod)                      # if allow_vectorize
mod = RewriteFp4ToFp4x2()(mod)                  # @Ascend: fp4 打包
mod = LoopUnswitching()(mod)
mod = UnrollLoop()(mod)
mod = RenormalizeSplitPattern()(mod)
mod = Simplify()(mod)
mod = AscendRemoveNoOp()(mod)
mod = HoistIfThenElse()(mod)
mod = Simplify()(mod)
mod = AscendRemoveNoOp()(mod)
mod = VerifyMemory()(mod)
mod = AnnotateEntryFunc()(mod)
mod = InferFragment()(mod)
mod = LowerThreadAllreduce()(mod)
mod = ThreadSync("global")(mod)                 # if allow_global_thread_sync
mod = AnnotateDeviceRegions()(mod)
mod = MarkScalarDcacheBypass()(mod)             # @Ascend: 标量 dcache bypass
mod = SplitHostDevice()(mod)
mod = AnnotateReadOnlyParams()(mod)
mod = MergeUBAllocations(align_bytes=32)(mod)   # @Ascend: ★ UB 合并 + reuse
mod = RewriteFlagToBuf()(mod)                   # @Ascend: ★ flag/buf 池优化
mod = ThreadSync("shared")(mod)
mod = ThreadSync("shared.dyn")(mod)
mod = MergeIfStmt()(mod)
mod = MakePackedAPI()(mod)
mod = Simplify()(mod)
mod = LowerDeviceKernelLaunch()(mod)
```

### 3.3.2 关键 Ascend 专用 Pass

| Pass | 作用 | 原理简述 | 调优实践 |
|------|------|---------|---------|
| **InsertNd2Nz** | ND→NZ 布局转换 | 自动检测 UB(ND)→L1(NZ) 的 T.copy，重写为 scatter + post_copy 序列 | → 第 5 篇 § 5.3.2 |
| **AutoSchedule (Z3)** | 自动流水线调度 | CSP 建模 + Z3 SMT 求解最小化 II | → 第 5 篇 § 5.6.1 |
| **MergeUBAllocations** | UB 合并 + reuse | 同步图活性分析 + 生命周期复用，align_bytes=32 | → 第 5 篇 § 5.6.3 |
| **RewriteFlagToBuf** | flag/buf 池优化 | 0/1 背包分配，在 8-slot flag 与 32-slot buf 池之间分配 | → 第 5 篇 § 5.4.3 |
| **AscendSimdVFLowerParallel** | SIMD 向量 lowering | SimdVF/SimtVF → 硬件向量指令 | → 第 5 篇 § 5.2 |
| **MarkScalarDcacheBypass** | 标量 dcache bypass | 标量 global 访问绕过 dcache | → 第 5 篇 § 5.6.4 |
| **RewriteFp4ToFp4x2** | fp4 打包优化 | fp4 → 1 字节 packed-pair，减半偏移/索引 | → 第 5 篇 § 5.6.4 |
| **RewriteAscendBufferVersionLayout** | 多缓冲 UB 32B 对齐 | 多版本 buffer 对齐优化 | → 第 5 篇 § 5.5 |
| **LegalizeSimdMerging** | SIMD merging 合法化 | MODE_MERGING 保留非活跃 lane，暴露 rw access | — |
| **UnrollLoopSkipVF** | VF 外 unroll | 仅展开 VF 块外的显式 unroll | — |
| **AscendInsertOOBPadding** | OOB padding | Clamp DMA copy 尾块 + GM→L1 padding fill | — |
| **NormalizeTask** | 任务归一化 | 将 T.Task/latency 标记归一化 | — |
| **LatencyEstimator** | latency 估计 | 估计各任务 latency 供 Z3 使用 | — |
| **WarpgroupPartition** | warpgroup 划分 | 划分 warpgroup | — |
| **RewriteDualCopy** | dual_copy 重写 | 重写 T.dual_copy 为具体指令 | — |

### 3.3.3 CCE 代码生成

```
target.build.tilelang_ascend
  → 生成 AscendC/CCE 源码（.cce）
  → bisheng.compile_ascend(cce_source)
  → .aibin（Ascend 二进制）
```

**intrinsics 映射**：

| TileLang API | CCE intrinsic | 说明 |
|-------------|---------------|------|
| `T.simd.vadd` | `Add(dst, src0, src1)` | 向量加 |
| `T.simd.vmul` | `Mul(dst, src0, src1)` | 向量乘 |
| `T.simd.vmax` | `Max(dst, src0, src1)` | 向量 max |
| `T.simd.vcvt` | `Cast(dst, src, ...)` | 类型转换 |
| `T.simd.vld` | `DataCopy(dst, src)` | 向量加载 |
| `T.copy(GM→UB)` | `DataCopy(ub, gm)` | MTE2 DMA |
| `T.copy(UB→GM)` | `DataCopy(gm, ub)` | MTE3 DMA |
| `T.gemm` | `Mmad(l0c, l0a, l0b)` | Cube GEMM |

**PTO 后端**：`target.build.tilelang_pto`（Pass-Time Optimization，编译期优化版）

---

## 3.4 CUDA 编译后端

### 3.4.1 CUDA Pass Pipeline

```python
# ════════════════════════════════════════════════════════════════
# 前段：Prologue
# ════════════════════════════════════════════════════════════════
mod = BindTarget(target)(mod)
mod = MaterializeKernelLaunch()(mod)
mod = LetInline()(mod)
mod = AddWrapperForSingleBufStore()(mod)
mod = LegalizeNegativeIndex()(mod)
mod = VerifyParallelLoop()(mod)
mod = InjectAssumes()(mod)
mod = Simplify()(mod)
mod = LayoutReducer()(mod)

# @CUDA-specific: Warp Specialization (Hopper TMA)
if allow_warp_specialized(target=target):
    mod = ProducerConsumerWarpSpecialized()(mod)  # ★ producer/consumer warp 特化

# @CUDA / Blackwell specific
mod = LowerBlackwell2SM()(mod)                     # ★ 2SM TCGEN5MMA

mod = IfStmtBinding()(mod)
mod = PipelinePlanning()(mod)
mod = InjectSoftwarePipeline()(mod)                # ★ 软件流水线注入
mod = Simplify()(mod)
mod = LayoutInference()(mod)
mod = LowerTileOp()(mod)

# @CUDA specific
mod = LowerL2Persistent()(mod)                     # ★ L2 持久化
mod = DecoupleTypeCast()(mod)
mod = LegalizeVectorizedLoop()(mod)
mod = LegalizeSafeMemoryAccess()(mod)
mod = LowerAccessPtr()(mod)
mod = Simplify()(mod)
mod = HoistNonRestrictParams()(mod)

# ════════════════════════════════════════════════════════════════
# 后段
# ════════════════════════════════════════════════════════════════
mod = LowerSharedTmem()(mod)                       # @CUDA: shared.tmem lowering
mod = PlanAndUpdateBufferAllocationLocation()(mod)
mod = LowerSharedBarrier()(mod)                    # @CUDA: mbarrier lowering
if has_tma:
    mod = FuseMBarrierArriveExpectTx()(mod)        # @CUDA: MBarrier 融合
mod = HoistGlobalBufferAllocations()(mod)
mod = LowerOpaqueBlock()(mod)
mod = Simplify()(mod)
mod = NarrowDataType(32)(mod)
mod = FlattenBuffer()(mod)
mod = ConfigIndexBitwidth()(mod)
mod = Simplify()(mod)
mod = VectorizeLoop()(mod)
mod = StorageRewrite()(mod)
mod = LoopUnswitching()(mod)
mod = UnrollLoop()(mod)
mod = RenormalizeSplitPattern()(mod)
mod = Simplify()(mod)
mod = RemoveNoOp()(mod)
mod = HoistIfThenElse()(mod)

# @CUDA specific: 后端 lowering
mod = LowerSharedTmem()(mod)
mod = LowerSharedBarrier()(mod)
mod = FuseMBarrierArriveExpectTx()(mod)
mod = LowerLDGSTG()(mod)                           # ★ ldg/stg lowering
mod = LowerHopperIntrin()(mod)                     # ★ Hopper intrinsic
mod = MergeSharedMemoryAllocations()(mod)          # ★ shared memory 合并
mod = InjectFenceProxy()(mod)                      # ★ fence proxy
mod = AnnotateWarpGroupRegAlloc()(mod)             # ★ 寄存器分配
mod = ... # 更多 lowering
mod = PersistThreadblock()(mod)                    # ★ 持久化 threadblock

mod = VerifyMemory()(mod)
mod = AnnotateEntryFunc()(mod)
mod = LowerThreadAllreduce()(mod)
mod = ThreadSync("shared")(mod)
mod = MakePackedAPI()(mod)
mod = LowerDeviceKernelLaunch()(mod)
```

### 3.4.2 关键 CUDA 专用 Pass

| Pass | 作用 | 硬件 | 原理 |
|------|------|------|------|
| **ProducerConsumerWarpSpecialized** | producer/consumer warp 特化 | Hopper TMA | 将 copy 和 compute 分配到不同 warp，利用 TMA 异步 |
| **LowerBlackwell2SM** | 2SM TCGEN5MMA | Blackwell | 跨 2 个 SM 的 MMA 指令 |
| **LowerL2Persistent** | L2 持久化 | 通用 | 重排 tile 访问顺序提升 L2 局部性 |
| **LowerSharedTmem** | shared.tmem lowering | Hopper | shared tensor memory lowering |
| **LowerSharedBarrier** | mbarrier lowering | Hopper | 硬件 mbarrier 代码生成 |
| **FuseMBarrierArriveExpectTx** | MBarrier 融合 | Hopper | 融合 arrive + expect_tx |
| **LowerLDGSTG** | ldg/stg lowering | 通用 | Ramp load/store → ldg/stg intrinsic |
| **LowerHopperIntrin** | Hopper intrinsic | Hopper | Hopper 专用 intrinsic lowering |
| **InjectFenceProxy** | fence proxy | 通用 | 插入 fence.proxy 同步 |
| **AnnotateWarpGroupRegAlloc** | 寄存器分配 | 通用 | warp 特化 set_max_nreg |
| **PersistThreadblock** | 持久化 threadblock | 通用 | grid-stride 循环 |

### 3.4.3 CUDA 编译命令

```python
# nvcc 编译
nvcc.compile_cuda(
    source_code,           # CUDA C++ 源码
    target_arch="sm_90",   # 目标架构
    options=[
        "--use_fast_math",
        "--ptxas-options=--register-usage-level=4",
        "-std=c++20",
    ],
)
# → 生成 ptx / cubin / fatbin
```

---

## 3.5 Z3 Auto-Schedule 深度解析（原理）

> 本节讲 Z3 调度器的原理和建模方法。调优实践（如何设置 latency、如何启用） → 详见第 5 篇 § 5.6.1。

### 3.5.1 问题建模

Z3 Auto-Schedule 将流水线调度建模为 **约束满足问题（CSP）**，用 Z3 SMT 求解器求解。

#### 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| `latencies[]` | list[int] | 每个任务的延迟（cycles） |
| `iis[]` | list[int] | 每个任务的 initiation interval（cycles） |
| `resource_flags[]` | list[int] | 资源 pipe mask（bitmask） |
| `data_deps[]` | list[(i, j, latency)] | 数据依赖：任务 j 依赖任务 i |
| `resource_deps[]` | list[(i, j)] | 资源依赖：任务 i 和 j 使用同一资源 |

**资源 bitmask**：

| bit | 引擎 | 值 |
|-----|------|---|
| 0 | MTE1 | 1 |
| 1 | MTE2 | 2 |
| 2 | MTE3 | 4 |
| 3 | Cube | 8 |
| 4 | Vector | 16 |
| 5 | Fixpipe | 32 |
| 6 | Scalar | 64 |

#### 决策变量

- `start_vars[i]`：每个任务的开始时间
- `O_ij`：布尔变量，任务 i 和 j 的排序（True = i before j）
- `makespan`：总完成时间
- `II`：initiation interval（循环流水线场景）

#### 约束

```python
# ① 非负约束
for var in start_vars:
    solver.add(var >= 0)

# ② 数据依赖约束：任务 j 必须在任务 i 开始 + latency 之后开始
for i, j, latency in data_deps:
    solver.add(start_vars[j] >= start_vars[i] + latency)

# ③ 资源依赖约束：同资源任务不能并行
for i, j in resource_deps:
    o_ij = z3.Bool(f"O_{i}_{j}")  # 排序变量
    # 若 o_ij=True (i before j): start_j >= start_i + ii_i
    solver.add(z3.Implies(o_ij, start_vars[j] >= start_vars[i] + iis[i]))
    # 若 o_ij=False (j before i): start_i >= start_j + ii_j
    solver.add(z3.Implies(z3.Not(o_ij), start_vars[i] >= start_vars[j] + iis[j]))

# ④ makespan 约束
for i in range(n):
    solver.add(makespan >= start_vars[i] + latencies[i])
```

#### 目标

```python
# 最小化 makespan
solver.minimize(makespan)
```

### 3.5.2 循环流水线调度

对于循环内的流水线调度，建模更复杂：

```python
def z3_schedule_loop_python(
    num_stages: int,           # 流水级数
    latencies: list[int],      # 各任务 latency
    iis: list[int],            # 各任务 II
    resource_flags: list[int], # 资源 mask
    data_deps: list[tuple],    # (i, j, distance, latency) 距离感知依赖
    resource_deps: list[tuple],
    buffer_sizes: list[int],   # 各 buffer 大小
    memory_groups: list[list], # [[capacity, idx0, idx1, ...], ...]
    ...
) -> tuple[list[int], list[int], int]:  # (start_times, sorted_indices, II)
```

**循环流水线建模**：

```
start_i = k_i * II + r_i
  其中 k_i = 迭代级数，r_i = 阶段内偏移

数据依赖（距离感知）：
  start_v - start_u >= latency_u - II * distance

资源依赖（模排序）：
  r_i - r_j + II * delta_ij >= ii_i
  r_i - r_j + II * (1 - delta_ij) >= ii_j
  其中 delta_ij 是布尔变量

buffer 容量约束：
  Sum(buffer_vars[i] * buffer_sizes[i]) <= capacity
```

**II 最小化**：二分搜索 `[max(iis), sum(latencies)+1)`

**线程安全**：`_z3_lock` 序列化 Z3 调用（Z3 非线程安全），每次独立 Context

### 3.5.3 用户调度提示 API

开发者可以通过以下 API 向 Z3 调度器提供提示：

| API | 作用 | 示例 |
|-----|------|------|
| `T.Task(latency=N, ii=M)` | 显式指定任务 latency/ii | `T.Task(latency=120, ii=1)` |
| `T.SimdVF(latency=N)` | 注入 SimdVF latency 标记 | `with T.SimdVF(latency=120):` |
| `T.SimtVF(latency=N)` | 注入 SimtVF latency 标记 | `with T.SimtVF(threads=128, latency=100):` |
| `T.PerCoreTask()` | 标记每核一个逻辑任务 | — |
| `T.assume_no_conflict(a, b)` | 声明不冲突，抑制保守同步 | `T.assume_no_conflict(buf_a, buf_b)` |

### 3.5.4 调度实例

以 GEMM 为例，5 个任务的调度：

```
任务列表：
  T0: MTE2(load A)    latency=100, resource=MTE2
  T1: MTE2(load B)    latency=100, resource=MTE2
  T2: Cube(gemm)      latency=80,  resource=Cube
  T3: FixPipe(convert) latency=20, resource=Fixpipe
  T4: MTE3(store)     latency=50,  resource=MTE3

数据依赖：
  T2 依赖 T0 (load A 完成)
  T2 依赖 T1 (load B 完成)
  T3 依赖 T2 (gemm 完成)
  T4 依赖 T3 (convert 完成)

资源依赖：
  T0 和 T1 同用 MTE2（不能并行）

Z3 求解结果：
  II = 2
  T0: start=0   (加载 frame 0 的 A)
  T1: start=100 (加载 frame 0 的 B，与 T0 串行)
  T2: start=200 (计算 frame 0，与下一轮 T0 重叠)
  T3: start=280 (转换 frame 0，与下一轮 T2 重叠)
  T4: start=300 (存储 frame 0，与下一轮 T3 重叠)

  → A/B 加载与上一轮 Cube 重叠
  → FixPipe 与下一轮 Cube 重叠
```

---

## 3.6 后端对比与跨平台设计

### 3.6.1 统一 pass 框架

TileLang 的编译后端采用**统一 pass 框架 + 后端专用 pass** 的设计：

```
tilelang/transform/           # 通用 pass（所有后端共享）
  ├── simplify.py
  ├── layout_inference.py
  ├── lower_tile_op.py
  └── ...

tilelang/ascend/              # Ascend 后端
  ├── pipeline.py             # AscendPassPipelineBody
  ├── codegen.py              # CCE codegen
  └── transform/              # Ascend 专用 pass
      ├── insert_nd2nz.py
      ├── auto_schedule.py    # Z3
      ├── merge_ub_allocations.py
      └── ...

tilelang/cuda/                # CUDA 后端
  ├── pipeline.py             # CUDAPassPipelineBody
  ├── codegen.py              # nvcc codegen
  └── transform/              # CUDA 专用 pass
      ├── warp_specialized.py
      ├── lower_hopper_intrin.py
      └── ...
```

**注册式分发**：

```python
# pipeline 注册
register_pipeline(PassPipeline("ascend", AscendPassPipelineBody))
register_pipeline(PassPipeline("cuda", CUDAPassPipelineBody))

# 分发
mod = resolve_pipeline(target)(mod)  # 根据 target.kind 选择 pipeline
```

### 3.6.2 codegen 注册式分发

```python
# codegen 注册
register_device_codegen("cuda", cuda_codegen, supports_target=lambda t: t.kind == "cuda")
register_device_codegen("ascend", ascend_codegen, supports_target=lambda t: t.kind == "c" and ...)
register_device_codegen("pto", pto_codegen, supports_target=lambda t: ...)

# 分发
device_code = resolve_device_codegen(target)(mod)
```

**同 target kind 多 codegen 共存**：如 `"c"` kind 下 Ascend (CCE) 与 PTO 并存。

### 3.6.3 Ascend vs CUDA pass 对比

| 功能 | Ascend Pass | CUDA Pass | 说明 |
|------|------------|-----------|------|
| 布局转换 | InsertNd2Nz | — | Ascend Cube 要求 NZ 布局 |
| 自动调度 | AutoSchedule (Z3) | PipelinePlanning + InjectSoftwarePipeline | Ascend 用 Z3 求解，CUDA 用启发式 |
| 内存合并 | MergeUBAllocations | MergeSharedMemoryAllocations | UB vs shared memory |
| 向量 lowering | AscendSimdVFLowerParallel | VectorizeLoop | SIMD vs 向量化 |
| 异步拷贝 | — | LowerPTXAsyncCopy / LowerLDGSTG | CUDA 有异步拷贝 |
| Warp 特化 | — | ProducerConsumerWarpSpecialized | Hopper TMA 特化 |
| FixPipe 重叠 | unit_flag_ctrl | — | Ascend 专有 FixPipe |
| 同步优化 | RewriteFlagToBuf | InjectFenceProxy | flag/buf vs fence |
| 寄存器控制 | — | AnnotateWarpGroupRegAlloc | CUDA warp 寄存器 |
| 持久化 | T.Persistent | PersistThreadblock | 两者都有持久化 |
| L2 优化 | l2_cache_ctrl | LowerL2Persistent | 不同策略 |
| 多缓冲 | AnnotateMultiBufferEligible | PipelinePlanning | 不同实现 |

### 3.6.4 架构差异总结

```
Ascend NPU 架构                    CUDA GPU 架构
┌──────────────────┐              ┌──────────────────┐
│ 6 级内存          │              │ 3 级内存          │
│ GM → UB → L1     │              │ GM → shared → reg │
│  → L0A/B → L0C   │              │                   │
│                   │              │                   │
│ 5 类引擎          │              │ SIMT 线程模型     │
│ MTE2/MTE3/Cube   │              │ warp + TMA        │
│ /Vector/Scalar   │              │                   │
│                   │              │                   │
│ Z3 auto-schedule │              │ 启发式流水线      │
│ NZ 布局约束       │              │ 无布局约束        │
│ FixPipe 后处理    │              │ 无 FixPipe        │
└──────────────────┘              └──────────────────┘
```

---

## 3.7 调试与性能分析

### 3.7.1 IR dump

```python
@tilelang.jit(
    target="ascend",
    pass_configs={
        PassConfigKey.TL_ENABLE_DUMP_IR: True,
        PassConfigKey.TL_DUMP_IR_DIR: "/tmp/ir_dump",
    },
)
def kernel(...):
    ...
```

dump 目录结构：

```
/tmp/ir_dump/
  ├── 00_input.py           # 原始 IR
  ├── 01_after_simplify.py  # Simplify 后
  ├── 02_after_layout.py    # LayoutInference 后
  ├── 03_after_nd2nz.py     # InsertNd2Nz 后
  ├── 04_after_autoschedule.py  # AutoSchedule 后
  ├── ...
  └── final.py              # 最终 IR
```

### 3.7.2 AST 打印

```python
PassConfigKey.TL_AST_PRINT_ENABLE: True
# 打印 AST 结构，用于调试 IR 变换
```

### 3.7.3 布局可视化

```python
PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True
# 可视化内存布局，用于调试 LayoutInference
```

### 3.7.4 Pass 耗时分析

```python
PassConfigKey.TL_PASS_PROFILE: True
PassConfigKey.TL_PASS_PROFILE_THRESHOLD_MS: 100  # 只显示 >100ms 的 pass
```

输出示例：

```
[PassProfile] AutoSchedule: 234ms
[PassProfile] MergeUBAllocations: 156ms
[PassProfile] LayoutInference: 89ms
[PassProfile] Simplify: 45ms
```

### 3.7.5 如何读 IR

```python
# dump 的 IR 是 TVM Script 格式，可直接阅读
# @tvm.script.ir_module
# class Module:
#     @T.prim_func
#     def main(A: T.Buffer((8192, 8192), "bfloat16"), ...):
#         # IR 内容
#         with T.Kernel(32) as bx:
#             shared = T.alloc_shared((256, 256), "bfloat16")
#             T.copy(A[bx*256:(bx+1)*256, :256], shared)
#             ...
```

**读 IR 的关键点**：
1. 看 `alloc_shared` / `alloc_l1` 的 shape 和 dtype → 确认内存分配
2. 看 `T.copy` 的 src/dst → 确认数据搬运路径
3. 看 `T.gemm` 的参数 → 确认 GEMM 配置
4. 看 `set_flag` / `wait_flag` → 确认同步关系
5. 看循环结构 → 确认流水深度

---

## 3.8 读者检查点

1. **从 @tilelang.jit 到 kernel 加载，经过哪些主要阶段？**
   - JITImpl → tilelang.lower → resolve_pipeline → pass pipeline → device_codegen → adapter → JITKernel

2. **Ascend pass pipeline 分为哪三段？AutoSchedule 在哪一段？**
   - 前段（规范化+布局）、Auto-Schedule 段（Z3）、后段（lowering+codegen）
   - AutoSchedule 在第二段

3. **Z3 auto-schedule 的决策变量和目标函数是什么？**
   - 决策变量：start_vars[i]（开始时间）、O_ij（排序）、makespan
   - 目标：minimize makespan

4. **MergeUBAllocations 和 MergeSharedMemoryAllocations 分别用于哪个后端？**
   - MergeUBAllocations → Ascend（UB 合并）
   - MergeSharedMemoryAllocations → CUDA（shared memory 合并）

5. **如何 dump 每轮 IR 来诊断编译问题？**
   - `TL_ENABLE_DUMP_IR=True` + `TL_DUMP_IR_DIR="/tmp/ir_dump"`

6. **Ascend 和 CUDA 的自动调度有什么区别？**
   - Ascend: Z3 SMT 求解器（精确最优）
   - CUDA: PipelinePlanning + InjectSoftwarePipeline（启发式）

---

## 3.9 小结

本篇深入解析了 TileLang 编译后端的完整架构：

| 层次 | 内容 |
|------|------|
| 编译流程 | @tilelang.jit → lower → pipeline → codegen → adapter |
| 通用 pass | 规范化 + 布局推断 + 循环优化 + 后端 lowering |
| Ascend pass | 40+ pass，三段式（前段 + Z3 调度 + 后段） |
| CUDA pass | 30+ pass，warp 特化 + Hopper/Blackwell lowering |
| Z3 auto-schedule | CSP 建模 + SMT 求解 + 循环流水线调度 |
| 后端对比 | 统一框架 + 后端专用 pass + 注册式分发 |

**核心结论**：
- TileLang 编译后端 = 统一 pass 框架 + 后端专用 pass + codegen 分发
- Ascend 后端核心：Z3 auto-schedule + UB merge + ND→NZ + SIMD lowering
- **理解编译后端是极致优化的基础**——知道 pass 做什么，才能知道如何调参

本篇讲了 pass 的 What 和 How；调优实践（Why/When） → 详见第 5 篇 § 5.6

下一篇将展示 6 类算子模板，将前两篇的 API 和编译知识付诸实践。
