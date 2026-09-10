"""
模板 B：GEMM / Cube 类算子

核心：GM → L1 → L0A/L0B → L0C → UB → GM 五级流水。
T.gemm 自动处理 L0→L0C 的 Cube 计算，用户负责 GM→L1 的 MTE2 搬运和 L0C→GM 的 FixPipe 写回。

适用算子：matmul、split-K GEMM、Linear
核心引擎：Cube + MTE2
参考文件：ascend/example_gemm.py
"""
import tilelang
import tilelang.language as T


def gemm(M_DIM=8192, K_DIM=8192, N_DIM=8192, dtype="bfloat16", out_dtype="float32"):
    """GEMM 算子 NPU 实现

    语义：C = X @ W^T
    输入：X [M, K], W [N, K]
    输出：C [M, N]
    """
    NUM_BLOCKS = 32
    TILE_M, TILE_N, TILE_K = 256, 256, 256  # bf16: TILE_K=256, fp32: TILE_K=128
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    NUM_STAGES = 2
    MIXED = (dtype != out_dtype)

    # aswt_swizzle macro：CUTLASS 风格 swizzle，提升 L2 cache 局部性
    WINDOW = 4
    MAIN_ROW = M_TILES // WINDOW
    TAIL_WIN = max(1, (OUT_TILES - MAIN_ROW * WINDOW * N_TILES) // N_TILES + 1)

    @T.macro
    def aswt_swizzle(tile_idx: T.Var("int32")) -> (T.Var("int32"), T.Var("int32")):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // N_TILES // WINDOW
        if row_idx < MAIN_ROW:
            m_tile = row_idx * WINDOW + tile_idx % WINDOW
            n_tile = (tile_idx // WINDOW) % N_TILES
        else:
            tail_idx = tile_idx - MAIN_ROW * WINDOW * N_TILES
            m_tile = MAIN_ROW * WINDOW + tail_idx % TAIL_WIN
            n_tile = (tail_idx // TAIL_WIN) % N_TILES
        if row_idx % 2 != 0:
            n_tile = N_TILES - 1 - n_tile  # 奇数行反向
        return m_tile, n_tile

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        }
    )
    def _kernel():
        @T.prim_func
        def main(
            X: T.Buffer((M_DIM, K_DIM), dtype),
            W: T.Buffer((N_DIM, K_DIM), dtype),  # 注意：[N, K] 布局
            C: T.Buffer((M_DIM, N_DIM), out_dtype),
        ):
            with T.Kernel(NUM_BLOCKS) as bx:
                # L0C 累加器 + L1 缓冲
                res = T.alloc_l0c((TILE_M, TILE_N), "float32")
                x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
                w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
                temp = T.alloc_shared((TILE_M // 2, TILE_N), out_dtype)

                # 持久化调度
                for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                    m_tile, n_tile = aswt_swizzle(tile_idx)

                    # K 维流水
                    for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                        # GM → L1（dn2nz 自动转换）
                        T.copy(
                            X[m_tile * TILE_M:(m_tile + 1) * TILE_M,
                              kt * TILE_K:(kt + 1) * TILE_K],
                            x_l1
                        )
                        T.copy(
                            W[n_tile * TILE_N:(n_tile + 1) * TILE_N,
                              kt * TILE_K:(kt + 1) * TILE_K],
                            w_l1
                        )
                        # L1 × L1 → L0C（Cube GEMM）
                        T.gemm(
                            x_l1, w_l1, res,
                            transpose_B=True,  # 强制约束
                            clear_accum=(kt == 0),
                            unit_flag_ctrl=T.Select(
                                kt == K_TILES - 1, T.UF_3, T.UF_2
                            )
                        )

                    # L0C → GM（FixPipe 输出）
                    if MIXED:
                        T.dual_copy(res, temp, unit_flag_ctrl=T.UF_3)
                        T.dual_copy(
                            temp,
                            C[m_tile * TILE_M:(m_tile + 1) * TILE_M,
                              n_tile * TILE_N:(n_tile + 1) * TILE_N]
                        )
                    else:
                        T.copy(
                            res,
                            C[m_tile * TILE_M:(m_tile + 1) * TILE_M,
                              n_tile * TILE_N:(n_tile + 1) * TILE_N]
                        )

        return main

    return _kernel()


"""
关键技术：

| 技术 | 说明 |
|------|------|
| aswt_swizzle | T.macro 地址 swizzle，提升 L2 cache 局部性 |
| transpose_B=True | 强制约束，权重 [N,K] 布局 |
| unit_flag_ctrl | 控制 FixPipe 与 GEMM 重叠，尾帧用 UF_3 |
| T.dual_copy | L0C→UB→GM 两步搬运，混合精度输出 |
| T.set_atomic("add") | 累加模式 C += A@B^T（Split-K 用） |
| T.set_hf32_mode | FP32 GEMM 的 HF32 模式 |

性能：Cube 利用率 80%+（bf16, TILE_K=256 + aswt_swizzle + unit_flag_ctrl）
"""
