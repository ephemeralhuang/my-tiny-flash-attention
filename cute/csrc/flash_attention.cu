#include "attention_api.cuh"
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/python.h>
#include <vector>

#include "static_switch.h"
#include "kernel_traits.h"
#include "flash.h"
#include "utils.h"

namespace flash
{
    using namespace cute;

    /// @param tensor: 输入张量，包含每个线程的 `tstS` 寄存器数据。张量的形状为 `(nrow=(2, MMA_M), ncol=(2, MMA_N))`，表示在自注意力计算中每个线程处理的矩阵片段。
    /// @param m_block: 当前块的行索引。用于计算张量在矩阵中的位置，帮助确定当前线程块应该处理的行范围。
    /// @param nbi: 当前块在列方向上的索引。帮助确定每个线程处理的列的偏移量。
    ///
    /// 该函数用于执行矩阵的 **屏蔽（masking）** 操作，特别是在矩阵乘法或自注意力机制中，
    /// 屏蔽掉不需要的部分，例如在因果模式下，确保每个位置只可以访问当前及之前的位置。
    ///
    /// - 屏蔽操作基于每个线程的 **行列索引**，通过比较列号和行号，决定哪些位置需要被屏蔽。
    /// - 该函数计算线程的 **处理范围**，并将超出范围的部分设置为 `-INFINITY`，使得在计算中忽略这些值。
    /// - 通常用于矩阵乘法（如 `QK^T` 计算）中，避免在自回归模型中泄露未来的信息。
    template <int kBlockM, int kBlockN, int kNWarps, typename Engine, typename Layout>
    inline __device__ void mask_within_nblock(Tensor<Engine, Layout> &tensor, const int m_block, const int nbi)
    {
        static_assert(Layout::rank == 2, "only support 2D Tensor");

        const int lane_id = threadIdx.x % 32;
        // 在SM80_16x8x16_F32F16F16F32_TN中，gemm的结果为(16,8),
        // 其中LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))，每个线程负责4个数，每一行都由四个线程的2个值组成
        const int col_idx_offset = nbi * kBlockN + (lane_id % 4) * 2;

        const int nrow_group = threadIdx.x / 32; // 总共128个线程， tiledmma：(64, 8 x2, 16)
        const int row_idx_offset = kBlockM * m_block + nrow_group * 16 + lane_id / 4;

        const int group_stride = kNWarps * 16; // tiledMMA M维度为64

#pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) // 负责for-loop(MMA_N) 表示(64,8)在 (BlockM,BlockN)中N维度tiling的次数
        {
            const int col_idx_base = col_idx_offset + nj * 8; // nj * 8是因为8是tiledMMA的N维度的大小
#pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); j++) // for-loop (2)
            {
                // j用于计算value 1和value 2对应col
                // col_idx最终表示当前thread所处理的value的列号
                const int col_idx = col_idx_base + j;

#pragma unroll
                for (int mi = 0; mi < size<0, 0>(tensor); mi++) // for-loop(2)
                {
/* code */
#pragma unroll
                    for (int i = 0; i < size<0, 1>(tensor); i++) // for-loop(MMA_M)
                    {
                        /* code */
                        const int row_idx = row_idx_offset + mi * 8 + i * group_stride;
                        if (col_idx > row_idx)
                        {
                            tensor(make_coord(mi, i), make_coord(j, nj)) = -INFINITY;
                        }
                    }
                }
            }
        }
    }
    /// @param acc: shape((2,2), 1, 8) if TiledMMA(SM80_16x8x16_F32F16F16F32_TN)=(64, 2X8, 16) and (AxB = C).shape=(64,64)
    /// @param tCrA: shape((2,2,2), 1, 4) if TiledMMA(SM80_16x8x16_F32F16F16F32_TN)=(64, 2X8, 16) and A.shape=(64,64)
    /// @param tCrB: shape((2,2), 8, 4) if TiledMMA(SM80_16x8x16_F32F16F16F32_TN)=(64, 2X8, 16) and B.shape=(64,64)
    template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
              typename TiledMMA, typename S2RTiledCopy, typename S2RThrCopy>
    inline __device__ void gemm_A_in_regs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 &tCsB,
                                          TiledMMA tiled_mma, S2RTiledCopy smem_tiled_copy_B,
                                          S2RThrCopy smem_thr_copy_B)
    {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));  // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));  // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K

        // 由于tCsB是由smem_tiled_copy_B对B进行tiling得到的tCsB的shape与tCrB不一致
        // 平铺 MMA 的目标寄存器张量的布局与源张量布局不立即兼容。
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        // 在k维度上进行传算交叠(communication compute overlap)的流水线，即做smem->reg拷贝的同时做gemm
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(tCrA); i++)
        {
            if (i < size<2>(tCrA) - 1)
            {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }

    template <typename Tensor0, typename Tensor1,
              typename Tensor2, typename Tensor3, typename Tensor4,
              typename TiledMMA, typename S2RTiledCopyA, typename S2RTiledCopyB,
              typename S2RThrCopyA, typename S2RThrCopyB>
    inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 &tCsA,
                                     Tensor4 &tCsB, TiledMMA tiled_mma, S2RTiledCopyA smem_tiled_copy_A,
                                     S2RTiledCopyB smem_tiled_copy_B, S2RThrCopyA smem_thr_copy_A, S2RThrCopyB smem_thr_copy_B)
    {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));  // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));  // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K

        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));

        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));

#pragma unroll
        for (int i = 0; i < size<2>(tCrA); i++)
        {
            if (i < size<2>(tCrA) - 1)
            {
                cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }

    template <int N>
    CUTE_HOST_DEVICE void cp_async_wait()
    {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
    }
    // Convert rowcol_layout from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16, or to ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
    template <typename MMA_traits, typename Layout>
    inline __device__ auto convert_layout_rowcal_Aregs(Layout rowcol_layout)
    {
        using X = Underscore;
        static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
        static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);

        constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});

        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        constexpr int MMA_N_Divisor = mma_shape_K == 8 ? 1 : 2;

        auto l = logical_divide(rowcol_layout, Shape<X, Shape<X, Int<MMA_N_Divisor>>>{}); // ((2, MMA_M), (2, (2, MMA_N / 2)))
        // TD [2023-08-13]: Same error as above on Cutlass 3.2
        // return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
        //                    get<0, 1>(l),
        //                    get<1, 1, 1>(l));

        return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
                           get<1>(get<0>(l)),
                           get<1>(get<1>(get<1>(l))));
    }

    template <typename Fragment>
    inline __device__ auto convert_type_f32_to_f16(Fragment const &acc_fp32)
    {
        Tensor acc_fp16 = make_tensor<cute::half_t>(shape(acc_fp32));
        {
            Tensor acc_fp32x2 = recast<float2>(acc_fp32);
            Tensor acc_fp16x2 = recast<__half2>(acc_fp16);
            for (int i = 0; i < size(acc_fp32x2); ++i)
            {
                acc_fp16x2(i) = __float22half2_rn(acc_fp32x2(i));
            }
        }
        return acc_fp16;
    }

    template <bool Scale_max = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale)
    {
        static_assert(Layout0::rank == 2, "Only support 2D Tensor");
        static_assert(Layout1::rank == 1, "Only support 1D Tensor");
        CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(max));

#pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++)
        {
            // If max is -inf, then all elements must have been -inf (possibly due to masking).
            // We don't want (-inf - (-inf)) since that would give NaN.
            // If we don't have float around M_LOG2E the multiplication is done in fp64.
            const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
#pragma unroll
            for (int ni = 0; ni < size<1>(tensor); ni++)
            {
                // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                // max * log_2(e)) This allows the compiler to use the ffma
                // instruction instead of fadd and fmul separately.
                tensor(mi, ni) = expf(tensor(mi, ni) * scale - max_scaled);
            }
        }
    }

    // Convert rowcol_layout from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16, or to ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
    template <typename MMA_traits, typename Layout>
    inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout)
    {
        using X = Underscore;
        static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
        static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
        constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
        auto l = logical_divide(rowcol_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{}); // ((2, MMA_M), (2, (2, MMA_N / 2)))
        // TD [2023-08-13]: Same error as above on Cutlass 3.2
        // return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
        //                    get<0, 1>(l),
        //                    get<1, 1, 1>(l));
        return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
                           get<1>(get<0>(l)),
                           get<1>(get<1>(get<1>(l))));
    };

    // Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    template <typename Layout>
    inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout)
    {
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{}); // ((2, 2), MMA_M, MMA_N)
        // TD [2023-08-13]: Idk why but get<0, 1>(l) doesn't work for Cutlass 3.2, I'm getting
        // "int_tuple.hpp(74): error: conversion to inaccessible base class"
        // return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
        return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
    };

    // scores:((2, MMA_M),(2, MMA_N))，经过了 causal 之后的 Q_i 和 k_j^T 的乘积，
    // scores_max:(2 * MMA_N), rowmax 的结果
    // scores_sum:(2 * MMA_M)， rowsum 的结果
    // acc_o:((2, 2),(MMA_M, MMA_N))， 最后的计算结果
    template <bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
    inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                             Tensor2 &acc_o, float softmax_scale_log2)
    {
        if (Is_first)
        {
            // NOTE: 第一次softmax不需要rescale, 只需要记录 Sij(kblockM, kblockN) 的 rowmax 和 rowsum
            reduce_max<true>(scores, scores_max);
            flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2); // scores --> scores_scaled
            reduce_sum(scores, scores_sum);
        }
        else
        {
            // 记录上一次的 mi(rowmax)
            Tensor scores_max_prev = make_fragment_like(scores_max);
            cute::copy(scores_max, scores_max_prev);

            // NOTE: 计算最新的 max
            // reduce_max包含步:
            //  1. 求当前thread内max: 遍历
            //  2. reduce thread间的max: 使用线程数洗牌指令做 all reduce，每个线程都获得了最大值
            reduce_max</*zero_init=*/false>(scores, scores_max); // scores_max 变成最新的最大值，相当于公式中的 m_i^{j}
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            // 将acc_o转换成符合2D直觉的(nrow, ncol)的形状
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));

#pragma unroll
            for (int mi = 0; mi < size(scores_max); mi++)
            {
                // NOTE: 辅助变量: 当前行max
                float scores_max_cur = scores_max(mi); // 当前行的最大值(未经过scaled)
                // NOTE: 计算上一次 score_sum 的 rescale 值
                float scores_scale = expf((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2); // 想当于公式中的 e^{m_i^{j-1} - m_i^{j}}.
                scores_sum(mi) *= scores_scale;                                                         // alpha * l_i{j-1}
#pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ni++)
                {
                    acc_o_rowcol(mi, ni) *= scores_scale;
                } // 相当于公式中的 e^{m_i^{j-1} - m_i^{j}}O_i^{j-1}
            }
            flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2); // P_i^_j

            Tensor scores_sum_cur = make_fragment_like(scores_sum);
            reduce_sum(scores, scores_sum_cur); // rowsum(p_i{j})
#pragma unroll
            for (int mi = 0; mi < size(scores_sum); mi++)
            {
                /* code */
                scores_sum(mi) += scores_sum_cur(mi); // l_i^{j} = alpha * l_i{j-1} + rowsum(p_i{j})
            }
        }
    }
}
void set_params_fprop(Flash_fwd_params &params,
                      const torch::Tensor q,
                      const torch::Tensor k,
                      const torch::Tensor v,
                      torch::Tensor out,
                      void *softmax_lse_d,
                      float softmax_scale,
                      bool is_causal)
{
    params.bs = q.size(0);
    params.head = q.size(1);
    params.q_seqlen = q.size(2);
    params.head_dim = q.size(3);

    params.k_head = k.size(1);
    params.k_seqlen = k.size(2);

    params.bs_stride = q.stride(0);
    params.head_stride = q.stride(1);
    params.seqlen_stride = q.stride(2);
    params.dim_stride = q.stride(3);

    params.softmax_scale = softmax_scale;

    params.softmax_scale_log2 = softmax_scale * M_LOG2E;
    params.is_causal = is_causal;
    params.is_bf16 = q.dtype() == torch::kBFloat16;

    params.softmax_lse_ptr = softmax_lse_d;

    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.out_ptr = out.data_ptr();
}

// Shared Storage with Aligned addresses.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage
{
    // TODO: Aligned的话smem的计算是否有问题
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

template <typename Kernel_traits, bool Is_causal = false, typename Params>
__global__ void flash_attention_v2_cutlass_kernel(const Params params)
{
    using namespace cute;

    // m block index
    const int m_block = blockIdx.x;
    // batch head
    const int batch_head_idx = blockIdx.y;

    const int tidx = threadIdx.x;
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using TiledMma = typename Kernel_traits::TiledMma;
    using index_t = typename Kernel_traits::index_t;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
    using SmemLayoutVtNswizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    // Shared memory
    extern __shared__ char smem_[];
    using SharedStorage = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_);

    const int bs_head_offset = batch_head_idx * params.head_stride;

    // NOTE: convert C pointer to Tensor for convenience
    Tensor Q = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + bs_head_offset),
        make_shape(params.q_seqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor K = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + bs_head_offset),
        make_shape(params.k_seqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor V = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + bs_head_offset),
        make_shape(params.k_seqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor O = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.out_ptr) + bs_head_offset),
        make_shape(params.q_seqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{}));

    // 加载Q, K, V分块
    // (kBlockM, kHeadDim, num_tile_n)
    Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

    // (kBlockN, kHeadDim, num_tile_n)
    // NOTE: loading流水线, 初次加载所需K, V
    Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    Tensor gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));

    // 获取TiledMMA
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);

    // Construct SMEM tensors.
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    // Tensor for V Transpose; used in GEMM-II.
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
    Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNswizzle{});

    // NOTE: copy抽象
    // NOTE: QKV gmem -> smem 拷贝的抽象
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    // NOTE: 定义gmem -> smem拷贝的src, dst
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0)); // (CPY, CPY_N, CPY_K) = ((1,8),4,1)
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // NOTE: 定义smem -> reg 拷贝的dst
    // partition_fragment与partition类似, 只是返回的是寄存器表示
    // sQ:(64, 64), TiledMMA:MK (64,16),根据 tensor core 的特性，可以推断出 tSrQ 的shape 为((2,2,2),1,4)
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    // sK:(64, 64), TiledMMA:NK:(16/2,16),根据 tensor core 的特性，可以推断出 tSrQ 的shape 为((2,2),8,4)
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    // NOTE: 准备拷贝Q, K 到 reg 的copy对象 (smem --> reg)

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    // 拷贝时转置
    // NOTE: 拷贝Vt smem->reg
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    flash::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);

    cute::cp_async_fence();

    Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{}); // （MMA, MMA_M, MMA_K) = ((2,2),1, 8)

    // step1: slice-k compute QK block
    // Q[BLOCK_M, BLOCK_N] @ K[BLOCK_M, BLOCK_N].T = O[BLOCK_M, BLOCK_M]
    //
    // step2:
    // advance K, V

    // NOTE: K, V分块的数量: 处理的区间
    const int n_block_min = 0;
    // NOTE: 1. mask between N BLOCKs if is causal mode
    int seqlen_start = m_block * kBlockM;
    int seqlen_end = (m_block + 1) * kBlockM;
    int n_block_max = Is_causal ? cute::ceil_div(seqlen_end, kBlockN) : cute::ceil_div(params.k_seqlen, kBlockN); // （2 * MMA_M）

    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});

    Tensor scores_sum = make_fragment_like(scores_max);

    clear(rAccOut);

    for (int nbi = n_block_min; nbi < n_block_max; nbi++)
    {
        auto rAccScore = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{})); // （MMA, MMA_M, MMA_N) ((2,2),1,8)
        clear(rAccScore);
        // 等待Q, K的gmem -> smem拷贝完成, 即Q, K就绪
        // wait<0>表示等待还剩0个未完成
        flash::cp_async_wait<0>();
        __syncthreads();

        // gemm的同时异步加载V
        gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
        tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));

        // 异步加载V到smem
        flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        // 发起异步拷贝
        cute::cp_async_fence();

        // O = Q@K.T
        // NOTE: 加载smem中的数据到reg再做gemm, **加载期间执行retile**
        flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                         smem_thr_copy_Q, smem_thr_copy_K);

        Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout())); // （MMA, MMA_M, MMA_N) --> ((2, MMA_M), (2, MMA_N))

        // NOTE: 2. mask within N BLOCKs
        if (Is_causal == true && nbi * kBlockN >= seqlen_start)
        {
            flash::mask_within_nblock<kBlockM, kBlockN, kNWarps>(scores, m_block, nbi);
        }

        // NOTE: 等待V加载完成, 为下个K加载准备初始状态
        flash::cp_async_wait<0>();
        __syncthreads();

        // advance K
        if (nbi != n_block_max - 1)
        {
            gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
            tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
            flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
            cute::cp_async_fence();
        }
        // 计算softmax
        // scores:((2, MMA_M),(2, MMA_N)), Q_i * K_j^T 的值
        // scores_max:(2 * MMA_N)
        // scores_sum:(2 * MMA_N)
        // rAccOut:((2, 2),(MMA_M, MMA_N))，相当于 O_i
        nbi == 0 ? flash::softmax_rescale_o</*Is_first=*/true>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale) : flash::softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale);

        // 计算完成后， scores 相当于公式中的 P_i^j
        // 实际执行 P_i^j @ V
        // (score AKA rAccScore): exp(QK[M, N] - m_i^j) @ V[N, dim]
        // NOTE: DABC: F32F16F16F32, convert D type(F32) to A type(F16)
        Tensor rP = flash::convert_type_f32_to_f16(rAccScore);

        // NOTE: Convert from layout C to layout A;  ((2, MMA_M),(2, MMA_N)) --> ((2, 2, 2),(MMA_M, MMA_N / 2))
        // ((2, 1), (2, 8)) ==> ((2, 2, 2), 1, 4)
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<TiledMma>(scores.layout()));
        // rAccOut:((2, 2),(MMA_M, MMA_N))
        flash::gemm_A_in_regs(rAccOut, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }
    // Epilogue
    // NOTE: 最后统一除上分母部分
    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    // AKA reshape to (nrow, ncol) but with specific MMA layout
    Tensor acc_o_rowcol = make_tensor(rAccOut.data(), flash::convert_layout_acc_rowcol(rAccOut.layout()));

// for row
#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi)
    {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        float scale = inv_sum;
// for col
#pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni)
        {
            acc_o_rowcol(mi, ni) *= scale;
        }
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type_f32_to_f16(rAccOut);
    // 复用sQ的smem做sO的拷出
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)

    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);    // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO); // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // NOTE: 先拷贝到smem
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

    // 创建到smem -> gmem的拷贝
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO); // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));

    __syncthreads();

    // NOTE:: 再拷贝到gmem
    cute::copy(gmem_tiled_copy_O, tOsO, tOgO);
}

template <typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream)
{
    using Element = typename Kernel_traits::Element;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

    const int num_m_block = (params.q_seqlen + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    // x = num_m_block y = bs * head, grid中的每一个block处理具体的一个batch中的一个head的（kBlockM, kHeadDim）
    dim3 grid(num_m_block, params.bs * params.head);
    dim3 block(Kernel_traits::kNThreads); // 一个block的线程

    int smem_size = int(sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));
    auto kernel = &flash_attention_v2_cutlass_kernel<Kernel_traits, Is_causal, Flash_fwd_params>;

    if (smem_size > 48 * 1024)
    {
        CUDA_ERROR_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
}

template <typename T, int HeadDim>
void run_flash_fwd_(Flash_fwd_params &params, cudaStream_t stream)
{
    BOOL_SWITCH(params.is_causal, Is_causal, [&]
                { run_flash_fwd<Flash_fwd_kernel_traits<HeadDim, /*kBlockM_=*/64, /*kBlockN_=*/64, /*kNWarps_=*/4, T>, Is_causal>(params, stream); });
}

// entry point of flash attention
void run_flash_attn_cutlass(Flash_fwd_params &params, cudaStream_t stream)
{
    // FP16_SWITCH yield elem_type namespace
    FP16_SWITCH(!params.is_bf16, [&]
                {
        // FWD_HEADDIM_SWITCH yield kHeadDim constexpr
        FWD_HEADDIM_SWITCH(params.head_dim, [&] {
            run_flash_fwd_<elem_type, kHeadDim>(params, stream);
        }); });
}

std::vector<torch::Tensor> flash_attention_v2_cutlass(torch::Tensor q, torch::Tensor k,
                                                      torch::Tensor v, bool is_causal = false, float softmax_scale = 1)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    int bs = q.size(0);     // batch_size
    int head = q.size(1);   // number of head
    int seqlen = q.size(2); // sequence length
    int dim = q.size(3);    // head dim

    auto out = torch::empty_like(q);

    Flash_fwd_params params;
    set_params_fprop(params, q, k, v, out,
                     nullptr, softmax_scale, is_causal);

    run_flash_attn_cutlass(params, 0);

    // wait until kernel finish
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK(cudaGetLastError());

    return {out};
}
