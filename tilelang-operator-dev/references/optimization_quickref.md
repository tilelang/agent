# 瓶颈→TileLang 优化速查表

> 本文件是 Step ⑧（瓶颈定位与优化）的核心参考。确认瓶颈类型后，查本表选择具体优化策略。

---

## 1. VEC Bound（Vector 标量读瓶颈）

**判定条件**：`aiv_vec_ratio` 最高，`aiv_vec_wait_ratio` > 95%，L2 read_hit_rate < 50%

| 序号 | 优化措施 | TileLang 实现 | 预期收益 | 参考文件 |
|------|---------|-------------|---------|---------|
| 1 | UB 融合 | `T.alloc_shared` 保留中间值，消除中间 GM 写回 | 1.2-1.5x | `vllm-ascend/softmax.py` |
| 2 | 寄存器复用 | `T.alloc_fragment` 缓存中间结果 | 1.3-2x | `ascend/example_rmsnorm.py` |
| 3 | 减少 Cast | 合并 dtype 转换，用 `S.vcvt` 一次完成 | 1.1-1.3x | — |
| 4 | 优化归约 | `T.alloc_reducer` + `T.finalize_reducer` | 1.2-1.5x | `ascend/example_rmsnorm.py` |
| 5 | 调整 SimtVF threads | `T.SimtVF(threads=N)` 匹配数据量 | 1.1-1.2x | — |

**实际案例**：RMSNorm fragment trick，消除 x 二次 UB 读，d=7168 达 88%+ 带宽

---

## 2. MTE2/MTE3 Bound（搬运带宽瓶颈）

**判定条件**：`ai*_mte2_ratio` 最高

| 序号 | 优化措施 | TileLang 实现 | 预期收益 | 参考文件 |
|------|---------|-------------|---------|---------|
| 1 | 减少 GM 往返 | 融合计算，中间结果留 UB/Fragment | 1.2-1.5x | — |
| 2 | 增大连续搬运粒度 | 增大 Tile（BD/BM/TILE_K） | 1.1-1.3x | `gather_opt.py` |
| 3 | 处理非对齐尾块 | `T.copy(..., pad_value=0)` | 消除越界 | `example_simdvf_topk_gate.py` |
| 4 | 重用常量/权重 | 保留在 UB/L1，避免重复 GM 读 | 1.1-1.3x | — |
| 5 | 搬运计算重叠 | `num_stages=N` 多缓冲流水 | 1.2-1.5x | — |
| 6 | 调整 L2 策略 | `T.copy(..., l2_cache_ctrl="NORMAL_FV")` | 1.1-1.2x | `example_gemm_bypass_l2.py` |

**实际案例**：gather d=128: SimtVF→T.copy DMA，296→87 us（3.4x），带宽 11%→39%

**硬件极限判定**：`aiv_mte2_ratio` > 95% 且带宽 > 60% → 接近 HBM 随机读极限，停止优化

---

## 3. CUBE Bound（GEMM 计算瓶颈）

**判定条件**：`aic_cube_ratio` 最高，`aic_mad_ratio` < 80%

| 序号 | 优化措施 | TileLang 实现 | 预期收益 | 参考文件 |
|------|---------|-------------|---------|---------|
| 1 | 调整矩阵 Tile | `TILE_M/N/K` 增大（bf16: K=256, fp32: K=128） | 1.1-1.3x | `example_gemm.py` |
| 2 | L1 数据复用 | `T.alloc_l1` 保留 K tile 数据 | 1.1-1.2x | — |
| 3 | L0 累加 | `T.alloc_l0c` + `clear_accum=(kt==0)` | 消除冗余写 | — |
| 4 | K 流水 | `T.Pipelined(K_TILES, num_stages=N)` | 1.2-1.5x | — |
| 5 | 持久化调度 | `T.Persistent` 工作窃取 | 1.1-1.2x | — |
| 6 | 输出路径 | `T.copy` / `T.dual_copy` 选择 | 1.1-1.2x | — |
| 7 | aswt_swizzle | T.macro 地址 swizzle 提升 L2 局部性 | 1.1-1.3x | `example_gemm.py:40` |

**实际案例**：GEMM TILE_K=256 (bf16) + aswt_swizzle，Cube 利用率 80%+

---

## 4. SCALAR Bound（标量循环开销瓶颈）

**判定条件**：`ai*_scalar_ratio` > 30%

| 序号 | 优化措施 | TileLang 实现 | 预期收益 | 参考文件 |
|------|---------|-------------|---------|---------|
| 1 | 编译期特化 | dtype/shape 积入 JIT 参数 | 1.2-2x | — |
| 2 | 移出循环不变量 | 提取到循环外 | 1.1-1.3x | — |
| 3 | 减少 T.serial | 改 `T.Parallel` 或 `T.SimtVF` 展平 | 1.3-2x | — |
| 4 | 减少动态标量访问 | 批量 T.copy 预加载到 UB | 1.2-1.5x | `gather_*_zos` |
| 5 | 调整 block 数 | 匹配 Tile 数，减少空转 | 1.1-1.2x | — |
| 6 | 合并小 Tile | 增大每 block 工作量 | 1.1-1.3x | — |

**实际案例**：gather d=656: 逐行→批量(BM=4)，745→449 us（1.7x）

---

## 5. 核间负载不均衡

**判定条件**：各核 Task Duration 差异 > 10%

| 序号 | 优化措施 | TileLang 实现 | 预期收益 |
|------|---------|-------------|---------|
| 1 | 检查 T.Kernel 匹配 | block 数 = Tile 数 | 1.1-1.3x |
| 2 | T.ceildiv 计算 Tile 数 | 确保尾块正确分配 | 消除空转 |
| 3 | 尾块分散到多 block | 避免单核处理大尾块 | 1.1-1.2x |
| 4 | T.Persistent 工作窃取 | `group_size=1` 自动均衡 | 1.1-1.3x |

---

## 6. Bank Conflict

**判定条件**：`aiv_vec_total_cflt_ratio` > 5%

| 序号 | 优化措施 | TileLang 实现 | 预期收益 |
|------|---------|-------------|---------|
| 1 | 调整 UB shape/padding | `T.alloc_shared` 加 padding | 1.1-1.3x |
| 2 | 改变行 stride | 调整 `T.Parallel` 索引映射 | 1.1-1.2x |
| 3 | 添加 padding 避免同周期命中 | UB 分配加 1 列 padding | 1.1-1.2x |

---

## 7. 流水重叠不足

**判定条件**：`vec + scalar + mte2 + mte3` ≈ 100%（完全串行）

| 序号 | 优化措施 | TileLang 实现 | 预期收益 | 参考文件 |
|------|---------|-------------|---------|---------|
| 1 | 自动多缓冲 | `T.annotate_buffer_versions({buf: N})` | 1.2-1.5x | `example_rmsnorm.py` |
| 2 | Pipelined 标记 | `annotations={"multi_buffer_eligible": [buf]}` | 1.2-1.5x | — |
| 3 | 手动 ping-pong | `T.annotate_manual_multi_buffer(buf)` | 1.2-1.5x | `example_manual_multibuffer.py` |
| 4 | 检查真实依赖 | 确认 RAW/WAR 依赖正确 | — | — |

---

## 8. L2 Cache 命中率低

**判定条件**：`ai*_total_hit_rate` < 50%

| 序号 | 优化措施 | TileLang 实现 | 预期收益 | 参考文件 |
|------|---------|-------------|---------|---------|
| 1 | Tile + 持久化调度 | `T.Persistent` 增加局部性 | 1.1-1.3x | — |
| 2 | L2 保留策略 | `T.copy(..., l2_cache_ctrl="NORMAL_FV")` | 1.1-1.2x | `example_gemm_bypass_l2.py` |
| 3 | 流式访问策略 | `NOTALLOC_*` 旁路 L2 | 1.1-1.2x | — |
| 4 | A/B 测试 | 对输入/权重/输出分别测试 | — | — |

---

## 9. FixPipe 与 GEMM 未重叠

**判定条件**：`aic_fixpipe_ratio` > 15% 且与 MAD 串行

| 序号 | 优化措施 | TileLang 实现 | 预期收益 | 参考文件 |
|------|---------|-------------|---------|---------|
| 1 | unit_flag_ctrl 重叠 | `unit_flag_ctrl=T.Select(kt==K_TILES-1, UF_3, UF_2)` | 1.1-1.2x | `example_gemm.py` |
| 2 | 调整输出 Tile | 减小 TILE_N 降低 FixPipe 压力 | 1.05-1.1x | — |
| 3 | 混合精度路径 | `T.dual_copy` L0C→UB→GM | — | — |

---

## 10. 头开销大

**判定条件**：头开销占比 > 30%

| 序号 | 优化措施 | TileLang 实现 | 预期收益 |
|------|---------|-------------|---------|
| 1 | 减少核数 | 小 shape 用更少核 | 1.1-1.3x |
| 2 | 简化 kernel 启动路径 | 减少动态分支 | 1.05-1.1x |
| 3 | 合并小 Tile | 增大每 block 工作量 | 1.1-1.2x |

---

## 交叉关联诊断

| 现象组合 | 根因假设 | TileLang 优先检查 |
|---------|---------|------------------|
| 高 vec_ratio + 高 bank conflict | UB 布局放大 Vector 耗时 | `T.alloc_shared` shape/padding、并行索引映射 |
| 高 mte2_time + 低 L2 hit rate | 数据复用或 L2 策略不合理 | Tile 调度、`T.Persistent`、`l2_cache_ctrl` |
| 高 fixpipe_ratio | 输出路径或地址对齐低效 | 输出 Tile、有效区间、`T.dual_copy` 路径 |
| 高 mte2 + 高 mte3 | GM 双向搬运饱和 | 融合中间结果、增大 Tile、减少 GM 往返 |
| 低 Block Dim + 高 Duration | 并行 Tile 数或 block 设置不足 | `T.Kernel` block 数、Tile shape |
| scalar 高 + 小 shape | 动态控制和启动占比高 | 编译期特化、减少 `T.serial` |
