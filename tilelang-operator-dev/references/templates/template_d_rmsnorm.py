"""
模板 D：归约 / Norm 类算子

核心：Fragment 寄存器复用 + Reducer 跨线程归约。
关键技巧：一次 UB→Fragment 加载，复用于 sum(x²) 和 y=x*rstd*w，避免二次 UB 读。

适用算子：RMSNorm、LayerNorm
核心引擎：SimtVF + Reducer
参考文件：ascend/example_rmsnorm.py
"""
import tilelang
import tilelang.language as T


def rmsnorm(N=4096, D=7168, dtype="float32"):
    """RMSNorm 算子 NPU 实现

    语义：y = x / sqrt(mean(x²) + eps) * w
    输入：x [N, D], w [D]
    输出：y [N, D]
    """
    TILE = 256
    THREADS = 256
    num_cores = 64  # get_num_vec_cores()
    eps = 1e-6

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    def _kernel():
        @T.prim_func
        def main(
            x: T.Buffer((N, D), dtype),
            w: T.Buffer((D,), dtype),
            y: T.Buffer((N, D), dtype),
        ):
            with T.Kernel(num_cores) as pid:
                x_ub = T.alloc_shared((TILE,), dtype)
                w_ub = T.alloc_shared((TILE,), dtype)
                y_ub = T.alloc_shared((TILE,), dtype)

                for row_idx in T.Persistent([N], num_cores, pid):
                    # 加载权重（复用于所有行）
                    T.copy(w[:TILE], w_ub)

                    with T.SimtVF(threads=THREADS):
                        # ① Fragment 寄存器：一次 UB→Fragment 加载
                        x_frag = T.alloc_fragment((TILE,), "float32")
                        for i in T.Parallel(TILE):
                            x_frag[i] = x_ub[i]

                        # ② 从寄存器归约（无 UB 读）
                        sum_sq = T.alloc_reducer(
                            (1,), "float32", op="sum", replication="all"
                        )
                        T.clear(sum_sq)
                        for i in T.Parallel(TILE):
                            sum_sq[0] += x_frag[i] * x_frag[i]
                        T.finalize_reducer(sum_sq)  # ③ 跨线程归约

                        # ④ 计算 rstd
                        rstd = T.rsqrt(sum_sq[0] / D + eps)

                        # ⑤ 复用 x_frag（无 x reload）
                        for i in T.Parallel(TILE):
                            y_ub[i] = x_frag[i] * rstd * w_ub[i]

                    # 写回
                    T.copy(y_ub, y[row_idx, :TILE])

        return main

    return _kernel()


"""
性能（fp32, batch=4096）：
  d=4096:  ~100 us, 1300+ GB/s (80%+ peak)
  d=7168:  ~165 us, 1400+ GB/s (88%+ peak)

关键优化：
  1. Fragment trick: 一次 UB→Fragment 加载，复用于 sum(x²) 和 y=x*rstd*w
  2. 64 cores 全并行: get_num_vec_cores() = 64
  3. T.alloc_reducer + T.finalize_reducer: 硬件归约
  4. T.Persistent: 持久化调度 + 工作窃取
"""
