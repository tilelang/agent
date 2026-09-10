"""
模板 E：量化 / Cast 类算子

核心：SimdVF + simd intrinsics 实现 amax 归约 + FP8 转换。
group_block 自适应（64/32/16/8），T.annotate_buffer_versions 多缓冲流水。

适用算子：per-token FP8 cast、per-block cast
核心引擎：SimdVF + simd
参考文件：ascend/example_simdvf_per_token_cast_to_fp8.py
"""
import tilelang
import tilelang.language as T
import tilelang.language.simd as S


def per_token_cast_to_fp8(M=4096, N=8192, dtype="bfloat16"):
    """per-token FP8 量化 NPU 实现

    语义：对每行 x，计算 amax = max(|x|)，scale = amax / 448，输出 fp8 = x / scale
    输入：x [M, N] bfloat16
    输出：x_fp8 [M, N/2] float8_e4m3 (packed), scale [M] float32
    """
    VL = 64  # SIMD 向量长度
    BM = 8  # 行批大小
    group_block = 64  # 自适应: 64/32/16/8
    num_cores = 64

    @tilelang.jit(
        out_idx=[1, 2],
        pass_configs={
            tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    def _kernel():
        @T.prim_func
        def main(
            x: T.Buffer((M, N), dtype),
            x_fp8: T.Buffer((M, N // 2), "float8_e4m3fn"),
            scale: T.Buffer((M,), "float32"),
        ):
            with T.Kernel(num_cores) as pid:
                y_ub = T.alloc_shared((BM, N), "float32")
                y_q_ub_fp8 = T.alloc_shared((BM, N // 2), "float8_e4m3fn")
                scale_ub = T.alloc_shared((BM,), "float32")

                # 多缓冲流水
                T.annotate_buffer_versions({y_ub: 4, y_q_ub_fp8: 4})

                for blk in T.Persistent([M // BM], num_cores, pid):
                    # 加载 x 到 UB
                    T.copy(x[blk * BM:(blk + 1) * BM, :], y_ub)

                    with T.SimdVF():
                        eps = S.vdup(1e-4, "float32")
                        fp8_max_reg = S.vdup(448.0, "float32")

                        for i in range(BM):
                            for j in range(N // (VL * 2)):
                                col = j * VL * 2

                                # ① 加载 64 个 float32
                                x0 = S.vld(y_ub[i, col])
                                x1 = S.vld(y_ub[i, col + 64])

                                # ② 跨 lane amax 归约
                                amax = S.vcmax(
                                    S.vmax(S.vabs(x0), S.vabs(x1))
                                )

                                # ③ 计算 scale
                                scale_val = S.vdiv(
                                    S.vmax(amax, eps), fp8_max_reg
                                )
                                scale_brc = S.vdupv(scale_val)  # 广播

                                # ④ 量化
                                q0 = S.vdiv(x0, scale_brc)
                                q1 = S.vdiv(x1, scale_brc)

                                # ⑤ FP8 转换
                                q0_fp8 = S.vcvt(q0, "float8_e4m3fn")
                                q1_fp8 = S.vcvt(q1, "float8_e4m3fn")

                                # ⑥ PK4 布局存储
                                S.vsts(
                                    y_q_ub_fp8[i, col // 2], q0_fp8,
                                    dist="PK4_B32"
                                )
                                S.vsts(
                                    y_q_ub_fp8[i, (col + 64) // 2], q1_fp8,
                                    dist="PK4_B32"
                                )

                                # 存 scale
                                S.vsts(scale_ub[i], scale_val)

                    # 写回
                    T.copy(y_q_ub_fp8, x_fp8[blk * BM:(blk + 1) * BM, :])
                    T.copy(scale_ub, scale[blk * BM:(blk + 1) * BM])

        return main

    return _kernel()


"""
优化记录（per_token_cast g32）：

| 轮次 | 优化策略 | 延迟(us) | 带宽 | 状态 |
|------|---------|---------|------|------|
| 基线 | 初始版本 | 201.9 | 49% | — |
| R1 | num_stages=3 | ↓ | ↑ | ✅ |
| R2 | UB-aware num_sf_slots | ↓ | ↑ | ✅ |
| R3 | PassConfig 调优 | ↓ | ↑ | ✅ |
| R4 | vand 替代 vabs | ↓ | ↑ | ✅ |
| R5 | 自适应 packed_row_block_k | ↓ | ↑ | ✅ |
| R6 | vld2+vcgmax bf16 归约 | 151.1 | 66% | ✅ |
| R7 | num_stages=4 g32 | 151.3 | 66% | ✅ |
| R8 | 5 个尝试 | — | — | ❌ 全部回退 |

最终: 1.34x 加速, MTE2 97% → HBM 带宽硬件极限
"""
