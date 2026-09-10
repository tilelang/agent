# msprof 8-CSV 字段定义与阈值参考

> 本文件是 Step ⑦（性能标准判定）的核心参考。按固定顺序读取 8 个 CSV，理解各字段含义和阈值。

---

## 1. OpBasicInfo.csv

**分析目标**：校验 kernel 名、频率、Task Duration、Block Dim

| 字段 | 说明 | 达标 | 警告 | 严重 |
|------|------|------|------|------|
| Op Name | kernel 名称 | 与 `--kernel-name` 精确匹配 | — | — |
| Task Duration (us) | 单次执行耗时 | — | — | — |
| Block Dim | 核间并行度 | 等于可用核数 | 远小于核数 | = 1 |
| Current Freq | 实际运行频率 | = Rated Freq | < Rated Freq | << Rated Freq |
| Rated Freq | 额定频率 | — | — | — |

**注意**：若 Current Freq < Rated Freq，说明芯片未满频，耗时数据不可靠。

---

## 2. PipeUtilization.csv

**分析目标**：确定主导流水、kernel_time、核间差异

| 字段 | 说明 | 达标 | 警告 | 严重 |
|------|------|------|------|------|
| aiv_vec_ratio | Vector 引擎占比 | 与算子类型匹配 | > 80% | > 90% 且无优化空间 |
| aiv_scalar_ratio | Scalar 引擎占比 | < 10% | 10-30% | > 30% |
| aiv_mte2_ratio | MTE2（读 DMA）占比 | < 30%（计算型） | 30-50% | > 50% |
| aiv_mte3_ratio | MTE3（写 DMA）占比 | < 30% | 30-50% | > 50% |
| aic_cube_ratio | Cube 引擎占比（AIC） | 40-70%（GEMM 型） | < 40% | < 20% |
| aic_mad_ratio | MAD 指令占比 | > 80%（GEMM 型） | 60-80% | < 60% |
| aic_fixpipe_ratio | FixPipe 占比 | < 5% | 5-15% | > 15% |

**流水重叠判据**：
- `vec + scalar + mte2 + mte3` ≈ 100% → 完全串行，需上双缓冲
- `vec + scalar + mte2 + mte3` > 130% → 流水重叠生效

**核间差异**：各核 Task Duration 差异 < 10% 为达标，> 30% 为严重不均衡。

---

## 3. ArithmeticUtilization.csv

**分析目标**：判断有效计算、Cast、指令结构

| 字段 | 说明 | 关注点 |
|------|------|--------|
| aic_cube_fops | Cube 浮点操作数 | 核算实际计算量 |
| aiv_vec_fp32_ratio | Vector FP32 指令占比 | 过高说明精度浪费 |
| aiv_vec_fops | Vector 浮点操作数 | 核算实际计算量 |
| aiv_vec_cast_ratio | Cast 指令占比 | 过高说明类型转换过多 |

---

## 4. Memory.csv

**分析目标**：计算总搬运量、带宽利用率、理论搬运时间

| 字段 | 说明 | 达标 | 警告 | 严重 |
|------|------|------|------|------|
| read_main_memory_datas | GM 读取总量 | — | — | — |
| write_main_memory_datas | GM 写入总量 | — | — | — |
| GM_to_UB_datas | GM→UB 搬运量 | — | — | — |
| GM_to_UB_bw_usage_rate | GM→UB 带宽利用率 | > 60% | 30-60% | < 30% |
| UB_to_GM_bw_usage_rate | UB→GM 带宽利用率 | > 60% | 30-60% | < 30% |

**带宽利用率基准**：
- > 60%：接近硬件极限，停止优化
- 30-60%：有优化空间
- < 30%：严重未利用，必然存在非 DMA 瓶颈

---

## 5. MemoryL0.csv

**分析目标**：CUBE kernel 的 L0A/L0B/L0C 带宽

| 字段 | 说明 | 关注点 |
|------|------|--------|
| aic_l0a_read_bw | L0A 读取带宽 | GEMM 操作数 A 的读取效率 |
| aic_l0b_read_bw | L0B 读取带宽 | GEMM 操作数 B 的读取效率 |
| aic_l0c_write_bw_cube | L0C 写入带宽 | GEMM 累加器写回效率 |

**注意**：非 Cube 算子（纯 Vector）的 L0 数据无意义。

---

## 6. MemoryUB.csv

**分析目标**：Vector kernel 的 UB 读写带宽

| 字段 | 说明 | 关注点 |
|------|------|--------|
| aiv_ub_read_bw_vector | Vector UB 读取带宽 | Vector 计算的数据供给效率 |
| aiv_ub_read_bw_scalar | Scalar UB 读取带宽 | 标量访问的 UB 效率 |
| aiv_ub_write_bw_vector | Vector UB 写入带宽 | Vector 计算结果写回效率 |

---

## 7. L2Cache.csv

**分析目标**：验证缓存命中假设

| 字段 | 说明 | 达标 | 警告 | 严重 |
|------|------|------|------|------|
| ai*_total_hit_rate | L2 总命中率 | > 80% | 50-80% | < 50% |
| ai*_read_hit_rate | L2 读命中率 | > 80% | 50-80% | < 50% |
| ai*_write_hit_rate | L2 写命中率 | > 80% | 50-80% | < 50% |

**优化方向**：
- 命中率低 → Tile + 持久化调度增加局部性
- 流式访问 → `l2_cache_ctrl` 旁路策略
- 对输入/权重/输出分别 A/B 测试不同策略

---

## 8. ResourceConflictRatio.csv

**分析目标**：Bank conflict、资源冲突、等待比例

| 字段 | 说明 | 达标 | 警告 | 严重 |
|------|------|------|------|------|
| aiv_vec_total_cflt_ratio | Vector 总冲突率 | < 5% | 5-15% | > 15% |
| aiv_vec_wait_ratio | Vector 等待比例 | < 10% | 10-30% | > 30% |
| aiv_mte2_wait_ratio | MTE2 等待比例 | < 10% | 10-30% | > 30% |
| aic_cube_wait_ratio | Cube 等待比例 | < 10% | 10-30% | > 30% |
| aic_mte2_wait_ratio | AIC MTE2 等待比例 | < 10% | 10-30% | > 30% |

**MTE2 等待判据**：`mte2_wait_ratio` > 95% 且 `mte2_ratio` > 95% → HBM 带宽硬件极限

---

## 理论耗时计算

### 搬运理论耗时

```
理论耗时(us) = 实际搬运数据量(Byte) / GM 峰值带宽
Ascend 950DT: GM 峰值带宽 ≈ 4 TB/s = 4000 GB/s
```

**MTE2/MTE3 带宽共享**：同时读写 GM 时，总带宽被共享：
```
理论耗时(us) = (MTE2搬运量 + MTE3搬运量) / GM带宽
```

### 计算理论耗时

```
理论耗时(us) = 实际计算量(FLOP) / 对应单元理论算力
```

| 计算单元 | 理论算力 |
|---------|---------|
| cube_fp16_bf16 | 432 TFLO0PS |
| cube_fp32 | 26 TFLOPS |
| cube_int8_fp8 | 864 TOPS |
| vector_fp32_add | 13.5 TFLOPS |
| vector_fp16_bf16_fma | 54 TFLOPS |

### 搬运量与计算量核算

**搬运量（GM↔UB）**：从源码枚举所有 GM 张量读写，逐个核算字节数。

| 方向 | 张量 | 字节计算 |
|------|------|---------|
| 读 | 每个输入 | 元素总数 × 单位元素字节数 |
| 写 | 每个输出 | 元素总数 × 单位元素字节数 |

**计算量（区分 Cube/Vector）**：
- **Cube**：`T.gemm` 路径 → 每 tile `2 × M × N × K` FLOP；无 gemm 则为 0
- **Vector**：`T.SimdVF` / `T.Parallel` 内逐元素运算，按「每元素运算次数 × 元素数」核算

```
有效带宽(GB/s)   = 总搬运字节 / 实测稳态耗时
带宽利用率       = 有效带宽 / GM 峰值带宽
实际算力(FLOP/s) = 实际计算量 / 实测稳态耗时
```

---

## 性能判定流程

```
读取 OpBasicInfo.csv → Task Duration, Block Dim
    ↓
读取 PipeUtilization.csv → 主导流水单元
    ↓
核算实际搬运数据量 + 实际运算数据量（区分 Cube/Vector）
    ↓
计算理论耗时（搬运量/带宽 或 计算量/算力）
    ↓
比较实际耗时 vs 理论耗时
    ├── 差距 < 20% → 性能达标（接近硬件极限）
    ├── 差距 20-50% → 有优化空间，查阅瓶颈优化表
    └── 差距 > 50% → 严重瓶颈，必须优化
```

---

## PipeTimeline 流水气泡分析

只分析关键路径核及最慢/最快核，记录：

| 分析项 | 说明 |
|--------|------|
| Scalar/MTE2/VEC/CUBE/MTE3 首尾时间 | 各流水的活跃区间 |
| MTE2↔计算 重叠时间与重叠率 | 读与算的并行度 |
| 计算↔MTE3 重叠时间与重叠率 | 算与写的并行度 |
| SetFlag/WaitFlag/barrier 等待区间 | 同步开销 |
| 头气泡 | 首条有效流水前的空闲时间 |
| 内部气泡 | 相邻指令簇间空闲时间 |
| 尾气泡 | 末条流水后的空闲时间 |

```
气泡率   = 流水空闲时间 / 关键路径时间
重叠率   = 两流水重叠时间 / 较短流水活跃时间
头尾开销 = Task Duration - 有效流水覆盖时间
```

**交付门禁**：必须计算关键核的头/内部/尾气泡、总气泡率和关键流水重叠率。
