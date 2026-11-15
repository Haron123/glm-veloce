#pragma once

// moe_vec_q (grouped MUL_MAT_ID, few tokens per expert tile) and its launcher. Only the units moe-vecq-<types>.cu include
// this file (explicit per-type instances compile in parallel); moe-vec.cu calls moe_vec_launch_type through
// moe-vec-common.cuh.

#include "moe-vec-common.cuh"
#include "vecdotq.cuh"

typedef float (*moe_vec_dot_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

template <ggml_type type> struct moe_vec_traits;
template <> struct moe_vec_traits<GGML_TYPE_Q4_0> { static constexpr int vdr = VDR_Q4_0_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q4_0_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q4_1> { static constexpr int vdr = VDR_Q4_1_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q4_1_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q5_0> { static constexpr int vdr = VDR_Q5_0_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q5_0_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q5_1> { static constexpr int vdr = VDR_Q5_1_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q5_1_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q8_0> { static constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q8_0_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q4_K> { static constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q4_K_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q5_K> { static constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q5_K_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q6_K> { static constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q6_K_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q2_K> { static constexpr int vdr = VDR_Q2_K_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q2_K_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_IQ2_XXS> { static constexpr int vdr = VDR_IQ2_XXS_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_iq2_xxs_q8_1; };

static constexpr int MOE_VEC_NWARPS = 4;

// T lanes share one row (T divides the warp); the block covers rows_per_block = nwarps*warp_size/T rows
template <ggml_type type, int NC, int T>
__launch_bounds__(MOE_VEC_NWARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void moe_vec_q(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ vy, const int2 * __restrict__ tiles,
        const int32_t * __restrict__ n_tiles, const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst, const int ncols_x, const int nrows_x, const int64_t stride_row_x, const int64_t stride_expert_x,
        const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst, const int64_t s2_dst) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = moe_vec_traits<type>::vdr;
    constexpr moe_vec_dot_t vec_dot = moe_vec_traits<type>::dot;
    constexpr int rows_per_block = MOE_VEC_NWARPS*warp_size/T;

    const int tile = blockIdx.y;
    if (tile >= *n_tiles) {
        return;
    }
    const int2 te    = tiles[tile];
    const int expert = te.x;
    const int col0   = te.y;
    const int ncols  = min(NC, expert_bounds[expert + 1] - col0);

    const int tid  = threadIdx.y*warp_size + threadIdx.x;
    const int lane = tid % T;
    const int row  = blockIdx.x*rows_per_block + tid / T;
    if (row >= nrows_x) {
        return; // whole T-lane groups exit together
    }

    const char * x = vx + expert*stride_expert_x;
    const int kbx_row = row*stride_row_x;

    const block_q8_1 * y[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        y[j] = vy + (col0 + min(j, ncols - 1))*stride_col_y; // idle columns repeat the last token
    }

    float tmp[NC] = {0.0f};

    const int blocks_per_row = ncols_x / qk;
    constexpr int items_per_block = qi/vdr;
    const int items = blocks_per_row*items_per_block;

    for (int it = lane; it < items; it += T) {
        const int kbx = it / items_per_block;
        const int kqs = vdr*(it % items_per_block);
        const int kby = kbx*(qk/QK8_1);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (j < ncols) { // uniform: skips the idle columns of a partial tile
                tmp[j] += vec_dot(x, y[j] + kby, kbx_row + kbx, kqs);
            }
        }
    }

    if constexpr (T > 1) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            tmp[j] = warp_reduce_sum<T>(tmp[j]);
        }
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        if (lane == j % T && j < ncols) {
            const int d = ids_dst[col0 + j];
            dst[(d / n_expert_used)*s2_dst + (d % n_expert_used)*s1_dst + row] = tmp[j];
        }
    }
}


template <ggml_type type, int NC>
static void moe_vec_launch_t(int T, dim3 grid_x_rows, const char * x, const block_q8_1 * y, const int2 * tiles, const int32_t * n_tiles,
        const int32_t * bounds, const int32_t * ids_dst, float * dst, int ncols_x, int nrows_x, int64_t s01, int64_t s02,
        int64_t s_col_y, int n_used, int64_t s1, int64_t s2, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const dim3 block(warp_size, MOE_VEC_NWARPS);
    const int rpb_1 = MOE_VEC_NWARPS*warp_size;
    auto go = [&](auto kern, int t) {
        dim3 grid((nrows_x + rpb_1/t - 1)/(rpb_1/t), grid_x_rows.y);
        kern<<<grid, block, 0, stream>>>(x, y, tiles, n_tiles, bounds, ids_dst, dst, ncols_x, nrows_x, s01, s02, s_col_y, n_used, s1, s2);
    };
    switch (T) {
        case 1:  go(moe_vec_q<type, NC, 1>,  1);  break;
        case 2:  go(moe_vec_q<type, NC, 2>,  2);  break;
        case 4:  go(moe_vec_q<type, NC, 4>,  4);  break;
        case 8:  go(moe_vec_q<type, NC, 8>,  8);  break;
        case 16: go(moe_vec_q<type, NC, 16>, 16); break;
        case 32: go(moe_vec_q<type, NC, 32>, 32); break;
        default: go(moe_vec_q<type, NC, 64>, 64); break;
    }
}

template <ggml_type type>
void moe_vec_launch_type(int NC, int T, dim3 grid, const char * x, const block_q8_1 * y, const int2 * tiles, const int32_t * n_tiles,
        const int32_t * bounds, const int32_t * ids_dst, float * dst, int ncols_x, int nrows_x, int64_t s01, int64_t s02,
        int64_t s_col_y, int n_used, int64_t s1, int64_t s2, cudaStream_t stream) {
    switch (NC) {
        case 4:  moe_vec_launch_t<type, 4> (T, grid, x, y, tiles, n_tiles, bounds, ids_dst, dst, ncols_x, nrows_x, s01, s02, s_col_y, n_used, s1, s2, stream); break;
        case 8:  moe_vec_launch_t<type, 8> (T, grid, x, y, tiles, n_tiles, bounds, ids_dst, dst, ncols_x, nrows_x, s01, s02, s_col_y, n_used, s1, s2, stream); break;
        default: moe_vec_launch_t<type, 16>(T, grid, x, y, tiles, n_tiles, bounds, ids_dst, dst, ncols_x, nrows_x, s01, s02, s_col_y, n_used, s1, s2, stream); break;
    }
}

#define MOE_VECQ_INSTANCE(t) \
    template void moe_vec_launch_type<t>(int, int, dim3, const char *, const block_q8_1 *, const int2 *, const int32_t *, \
        const int32_t *, const int32_t *, float *, int, int, int64_t, int64_t, int64_t, int, int64_t, int64_t, cudaStream_t);
