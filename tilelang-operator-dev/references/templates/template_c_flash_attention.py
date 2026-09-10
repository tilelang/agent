"""
模板 C：Flash Attention（Cube + Vector 协作）

核心：Cube 负责 QK^T 和 PV 两次 GEMM，Vector (SimdVF) 负责 online softmax。
难点：softmax 输出需要以 NZ 布局直接写入 UB，供第二次 GEMM 消费，避免 ND→NZ 转换。

适用算子：MHA、GQA 前向
核心引擎：Cube + SimdVF
参考文件：ascend/flash_attention/core.py
"""
import tilelang
import tilelang.language as T
import tilelang.language.simd as S


def flash_attention_forward(
    BATCH=1, HEADS=32, SEQ=128, D_HEAD=128,
    dtype="bfloat16", out_dtype="float32"
):
    """Flash Attention 前向 NPU 实现

    语义：O = softmax(Q @ K^T / sqrt(d)) @ V
    输入：Q [B, H, S, D], K [B, H, S, D], V [B, H, S, D]
    输出：O [B, H, S, D]
    """
    BR = 64  # Query block size
    BC = 128  # KV block size (= 2 * VL)
    NUM_STAGES = 3
    NUM_CORES = 32  # cube cores

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
            Q: T.Buffer((BATCH, HEADS, SEQ, D_HEAD), dtype),
            K: T.Buffer((BATCH, HEADS, SEQ, D_HEAD), dtype),
            V: T.Buffer((BATCH, HEADS, SEQ, D_HEAD), dtype),
            O: T.Buffer((BATCH, HEADS, SEQ, D_HEAD), out_dtype),
        ):
            with T.Kernel(NUM_CORES) as pid:
                # 分配各级 buffer
                Q_shared = T.alloc_shared((BR, D_HEAD), dtype)
                K_shared = T.alloc_shared((BC, D_HEAD), dtype)
                V_shared = T.alloc_shared((BC, D_HEAD), dtype)

                qk_acc_l0c = T.alloc_l0c((BR, BC), "float32")
                pv_acc_l0c = T.alloc_l0c((BR, D_HEAD), "float32")

                S_ub = T.alloc_shared((BR, BC), "float32")
                P_nz_ub = T.alloc_shared((BR, BC), dtype)  # NZ 布局
                O_ub = T.alloc_shared((BR, D_HEAD), "float32")

                m_ub = T.alloc_shared((BR,), "float32")  # running max
                l_ub = T.alloc_shared((BR,), "float32")  # running sum
                alpha_ub = T.alloc_shared((BR,), "float32")  # scale factor

                # 每个 KV tile 内执行 5 步
                for kv in T.Pipelined(SEQ // BC, num_stages=NUM_STAGES):
                    # ① Cube: QK^T → L0C → UB
                    T.copy(Q[...], Q_shared)
                    T.copy(K[...], K_shared)
                    T.gemm(Q_shared, K_shared, qk_acc_l0c,
                           transpose_B=True, clear_accum=(kv == 0))
                    T.copy(qk_acc_l0c, S_ub)

                    # ② SimdVF: online softmax + NZ packing
                    with T.SimdVF():
                        # NZ 布局标注
                        T.annotate_layout({
                            P_nz_ub: T.make_ascend_compact_nz_layout(P_nz_ub)
                        })
                        for i in range(BR):
                            # online softmax: m_new = max(m_old, rowmax(S))
                            # alpha = exp(m_old - m_new), l = alpha * l_old + rowsum(P)
                            row_max = S.vcmax(S.vld(S_ub[i, :]))
                            # exp(x - max) 融合，避免 overflow
                            S.vexpdif(S.vld(S_ub[i, :]), row_max, ...)
                            # NZ packing: vcvf 分离偶数/奇数位，vor 合并
                            even = S.vcvt(e, "bfloat16", part=0)
                            odd = S.vcvt(e, "bfloat16", part=1)
                            packed = S.vor(even, odd)
                            # NZ 布局存储
                            S.vsstb(ptr, packed, stride=..., update=True)

                    # ③ L0C→UB NZ→ND 转换
                    T.copy(P_nz_ub, P_shared)  # NZ → ND

                    # ④ Cube: P@V → L0C → UB
                    T.copy(V[...], V_shared)
                    T.gemm(P_shared, V_shared, pv_acc_l0c,
                           clear_accum=(kv == 0))
                    T.copy(pv_acc_l0c, O_tmp_ub)

                    # ⑤ SimdVF: O = alpha*O + O_tmp
                    with T.SimdVF():
                        for i in range(BR):
                            alpha = S.vld(alpha_ub[i])
                            o_old = S.vld(O_ub[i, :])
                            o_new = S.vadd(S.vmul(alpha, o_old),
                                           S.vld(O_tmp_ub[i, :]))
                            S.vsts(O_ub[i, :], o_new)

                # 写回最终输出
                T.copy(O_ub, O[...])

        return main

    return _kernel()


"""
NZ 布局 softmax packing 关键 API：

| API | 说明 |
|-----|------|
| T.annotate_layout({buf: make_ascend_compact_nz_layout(buf)}) | 指定 NZ 布局 |
| S.vcvt(e, "bfloat16", part=0/1) | 分离偶数/奇数位 |
| S.vor(even, odd) | 合并为 bf16x128 |
| S.vsstb(ptr, data, stride=..., update=True) | NZ 布局存储 |
| S.vexpdif(x, max) | 融合 exp(x-max) 避免 overflow |

当前限制：NZ packing 路径硬编码 head_dim == 128, BR % 2 == 0, BC == 2 * VL (128)
"""
