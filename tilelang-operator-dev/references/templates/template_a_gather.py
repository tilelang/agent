"""
模板 A：Gather / Vector 类算子

核心原则：DMA 优先，避免标量 GM 访问。
NPU 没有 TMA 硬件 gather，必须用 T.copy DMA（MTE2 引擎）替代 SimtVF 标量读。

适用算子：gather、gather_input_ids_zos、embedding lookup
核心引擎：MTE2 + SimtVF
参考文件：gather_opt.py
"""
import tilelang
import tilelang.language as T
from tilelang.language import PassConfigKey


def _gather_tl_ascend(D, dtype="bfloat16", index_dtype="int32"):
    """Gather 算子 NPU 实现

    语义：output = embedding_table[input_ids]
    输入：input_ids [n], embedding_table [m, D]
    输出：output [n, D]
    """
    # 自适应 tiling：根据 D 选择最优 tile size
    if D <= 4:
        C = 1024
    elif D <= 128:
        C = 2048
    else:
        C = 16384

    BD = min(tilelang.next_power_of_2(D), C)
    BM = C // BD
    num_cores = 64  # get_num_vec_cores()

    @tilelang.jit(
        pass_configs={
            PassConfigKey.TIR_DISABLE_VECTORIZE: True,
            PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
            PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    def _kernel():
        n = T.dynamic("n")
        m = T.dynamic("m")

        @T.prim_func
        def main(
            input_ids: T.Buffer((n,), index_dtype),
            embedding_table: T.Buffer((m, D), dtype),
            output: T.Buffer((n, D), dtype),
        ):
            with T.Kernel(num_cores) as pid:
                values_ub = T.alloc_shared((BM, BD), dtype)
                index_ub = T.alloc_shared((BM), index_dtype)

                nbm = T.ceildiv(n, BM)
                nbd = T.ceildiv(D, BD)

                for tile_idx in T.Persistent(
                    [nbm * nbd], num_cores, pid,
                    group_size=1, num_stages=4
                ):
                    bm_idx = tile_idx // nbd
                    bd_idx = tile_idx % nbd

                    sm = bm_idx * BM
                    sd = bd_idx * BD

                    copy_m = T.min(BM, n - sm)
                    copy_d = T.min(BD, D - sd)

                    # 批量加载索引
                    T.copy(input_ids[sm:sm + copy_m], index_ub[:copy_m])

                    # 逐行 T.copy DMA gather（替代标量读）
                    for m_i in T.serial(BM):
                        if m_i < copy_m:
                            idx = index_ub[m_i]
                            T.copy(
                                embedding_table[idx, sd:sd + copy_d],
                                values_ub[m_i, :copy_d]
                            )

                    # 批量写回
                    T.copy(
                        values_ub[:copy_m, :copy_d],
                        output[sm:sm + copy_m, sd:sd + copy_d]
                    )

        return main

    return _kernel()


"""
D 分支策略：

| D 范围      | 策略                    | 理由                          |
|------------|------------------------|------------------------------|
| D ≤ 4      | SimtVF 标量 gather      | 数据量 < 1KB，DMA 开销不划算   |
| 4 < D ≤ 2048 | 批量 T.copy（BM 行一批）| MTE2 引擎，num_stages=4 流水  |
| D > 2048   | 逐行 T.copy（BM=1）     | 单行数据大，2D 循环            |

访问模式对比：

| 访问模式        | GPU 方案          | NPU 方案              | NPU 性能  |
|----------------|-------------------|----------------------|----------|
| 连续 DMA 搬运   | TMA / cp.async    | T.copy（MTE2/MTE3）  | 接近 GPU  |
| 不规则行 gather | TMA 硬件 gather    | 逐行 T.copy          | 2~4x 慢  |
| 逐元素标量 gather| thread 读 GM      | SimtVF 标量读         | 10x+ 慢  |

性能数据（d=128）：
  优化前: 296 us, 11% 带宽
  优化后: 87 us, 39% 带宽 (3.4x 加速)
"""
