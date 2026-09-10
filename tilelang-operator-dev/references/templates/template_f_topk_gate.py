"""
模板 F：TopK / 选择类算子

核心：SimdVF 选择排序循环，k 次迭代每次选出最大值。
T.copy(..., pad_value=-inf) 处理非 32B 对齐尾部；NUM_STAGES=6 深流水。

适用算子：MoE topk gate、标量 topk
核心引擎：SimdVF + simd
参考文件：ascend/example_simdvf_topk_gate.py
"""
import tilelang
import tilelang.language as T
import tilelang.language.simd as S


def moe_topk_gate(
    M=4096, N=256, K=8,
    dtype="float32"
):
    """MoE TopK Gate 算子 NPU 实现

    语义：对每行 scores，选出 top-K 个最大值及其索引
    输入：scores [M, N]
    输出：topk_values [M, K], topk_indices [M, K]
    """
    VL = 64  # SIMD 向量长度
    NUM_VREGS = 4  # 4 个 vreg 覆盖 256 lane (= N)
    NUM_STAGES = 6  # 深流水
    num_cores = 64

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
            scores: T.Buffer((M, N), dtype),
            topk_values: T.Buffer((M, K), dtype),
            topk_indices: T.Buffer((M, K), "int32"),
        ):
            with T.Kernel(num_cores) as pid:
                s_ub = T.alloc_shared((N,), dtype)
                out_ub = T.alloc_shared((K,), dtype)
                idx_ub = T.alloc_shared((K,), "int32")

                T.annotate_buffer_versions({s_ub: NUM_STAGES, out_ub: NUM_STAGES})

                for row in T.Persistent([M], num_cores, pid):
                    # 加载 scores，pad_value 处理非对齐尾部
                    T.copy(scores[row, :N], s_ub, pad_value=float('-inf'))

                    with T.SimdVF():
                        # 4 个 vreg 覆盖 256 lane
                        v = [S.vld(s_ub[i * VL]) for i in range(NUM_VREGS)]
                        r = [S.vci(i * VL) for i in range(NUM_VREGS)]  # 对应 index
                        neg_inf = S.vdup(float('-inf'), dtype)
                        one = S.vdup(1.0, "float32")

                        for k in range(K):
                            # ① 全局 max
                            mx = S.vdupv(
                                S.vcmax(
                                    S.vmax(
                                        S.vmax(v[0], v[1]),
                                        S.vmax(v[2], v[3])
                                    )
                                )
                            )

                            # ② 最小 index（tie-break）
                            idx = S.vdupv(
                                S.vcmin(
                                    S.vsel(
                                        S.vci(0x7FFFFFFF),
                                        S.vci(0),
                                        S.vcmp(S.vsel(v[0], v[1], S.vcmp(v[0], v[1], "ge")),
                                               S.vsel(v[2], v[3], S.vcmp(v[2], v[3], "ge")),
                                               S.vcmp(S.vsel(v[0], v[1], S.vcmp(v[0], v[1], "ge")),
                                                      S.vsel(v[2], v[3], S.vcmp(v[2], v[3], "ge")),
                                                      "ge"))
                                    )
                                )
                            )

                            # ③ 存 winner
                            S.vsts(out_ub[k], mx, one, "ONEPT_B32")
                            S.vsts(idx_ub[k], idx, one, "ONEPT_B32")

                            # ④ mask out winner
                            for i in range(NUM_VREGS):
                                v[i] = S.vsel(
                                    neg_inf, v[i],
                                    S.vcmp(r[i], idx, "eq")
                                )

                    # 写回
                    T.copy(out_ub, topk_values[row, :K])
                    T.copy(idx_ub, topk_indices[row, :K])

        return main

    return _kernel()


"""
关键 API：

| API | 说明 |
|-----|------|
| S.vld / S.vsts | 向量加载/存储 |
| S.vmax / S.vcmax | 逐元素 max / 跨 lane max 归约 |
| S.vmin / S.vcmin | 逐元素 min / 跨 lane min 归约（tie-break） |
| S.vcmp | 条件比较 |
| S.vsel | 条件选择 |
| S.vci | 创建常量 index 向量 |
| S.vdupv | 广播标量到所有 lane |
| T.copy(..., pad_value=-inf) | 处理非 32B 对齐尾部 |

性能：NUM_STAGES=6 深流水，M=4096 N=256 K=8
"""
