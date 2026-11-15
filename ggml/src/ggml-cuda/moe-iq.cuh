#pragma once

// i-quant MUL_MAT_ID decode kernels (moe_coal_iq, moe_coal_iq_flat) and their launchers. Only the per-type units
// moe-iq-<type>.cu include this file (one type per unit, so the ~600 kernel variants compile in parallel); moe-vec.cu
// calls the launchers through moe-vec-common.cuh.

#include "moe-vec-common.cuh"
#include "vecdotq.cuh"

#include <algorithm>


// ---- i-quant decode (iq2_xs / iq3_xxs), direct per slot ----
// GLM-5.3-Flash UD-Q2_K_XL: gate/up experts iq2_xs, down iq3_xxs (MMVQ took them before: tg 41 t/s at TP4 vs 57 for
// the larger Q4_K_XL). moe_coal_kq_ls with one 32-value sub-block per item: LPR lanes per row walk the row's sub-blocks
// (adjacent lanes, adjacent sub-blocks), the activation row sits in LDS. Signs are applied
// bytewise as in moe_rowlane_iq2xxs (every grid byte is > 0, so (g ^ m) + (m & 1) negates without carries); the
// sub-block sums are scaled in integer math (iq2_xs: (2*ls + 1) per 16 values, / 8; iq3_xxs: (2*ls + 1), / 4), which is
// the arithmetic of vec_dot_iq2_xs_q8_1 / vec_dot_iq3_xxs_q8_1 without their integer rounding.
static __device__ __forceinline__ uint32_t moe_iq_u16x2(const char * p) { // 4 bytes at a 2-byte aligned address
    return (uint32_t) *(const uint16_t *) p | ((uint32_t) *(const uint16_t *) (p + 2) << 16);
}

// one item's weight bytes (loaded ahead of their use): iq2_xs: d, qs of the sub-block (4 x 16 bits), its scale byte;
// iq3_xxs: d, the 8 grid indices, the signs | scale word; iq4_xs: d, the sub-block's 16 qs bytes (a, b, c, e), its
// 6-bit scale - 32 (s)
struct moe_iq_raw {
    uint32_t a, b, c, e;
    uint16_t d;
    int      s;
};

template <ggml_type type>
static __device__ __forceinline__ moe_iq_raw moe_iq_load(const char * __restrict__ row, const int it) {
    const int sb = it / 8, ib = it % 8;
    moe_iq_raw r;
    if constexpr (type == GGML_TYPE_IQ2_XS) {
        // 74 bytes: d, qs[32] (9-bit grid index | 7-bit sign index << 9), scales[8] (4 bits per 16 values)
        const char * b = row + sb*sizeof(block_iq2_xs);
        r.d = *(const uint16_t *) b;
        r.a = moe_iq_u16x2(b + 2 + 8*ib);
        r.b = moe_iq_u16x2(b + 6 + 8*ib);
        r.c = (uint8_t) b[2 + QK_K/4 + ib];
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        // 136 bytes: d, scales_h (2 bits per sub-block), scales_l[4] (4 bits per sub-block), qs[128] (16 bytes per
        // sub-block: value j in the low nibble of byte j, value 16 + j in its high nibble); rows are 8-byte aligned
        const char * b = row + sb*sizeof(block_iq4_xs);
        const uint32_t h = *(const uint32_t *) b;
        const uint2 q0 = *(const uint2 *) (b + 8 + 16*ib);
        const uint2 q1 = *(const uint2 *) (b + 16 + 16*ib);
        r.d = h & 0xFFFF;
        r.s = (int) ((((uint8_t) b[4 + ib/2] >> (4*(ib & 1))) & 0xF) | (((h >> (16 + 2*ib)) & 3) << 4)) - 32;
        r.a = q0.x; r.b = q0.y; r.c = q1.x; r.e = q1.y;
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        // 66 bytes: d, qs[32] (per sub-block: 4 grid indices (8 bits), then 4 x 7-bit sign sets | 4-bit scale << 28)
        const char * b = row + sb*sizeof(block_iq2_xxs);
        r.d = *(const uint16_t *) b;
        r.a = moe_iq_u16x2(b + 2 + 8*ib);
        r.b = moe_iq_u16x2(b + 6 + 8*ib);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        // 18 bytes per 32 values: d, qs[16] (value j in the low nibble of byte j, value 16 + j in its high nibble);
        // 2-byte-safe loads (18-byte blocks are never 4-byte aligned)
        const char * b = row + it*sizeof(block_iq4_nl);
        r.d = *(const uint16_t *) b;
        r.a = moe_iq_u16x2(b + 2);
        r.b = moe_iq_u16x2(b + 6);
        r.c = moe_iq_u16x2(b + 10);
        r.e = moe_iq_u16x2(b + 14);
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        // 18 bytes per 64 values (this fork's QK2_0 = 64): d, qs[16]; item it is chunk it%2 of block it/2
        const char * b = row + (it/2)*sizeof(block_q2_0);
        r.d = *(const uint16_t *) b;
        r.a = moe_iq_u16x2(b + 2 + 8*(it % 2));
        r.b = moe_iq_u16x2(b + 6 + 8*(it % 2));
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        // 82 bytes per 256: d, qs[32] grid indices, qs[32..63] sign bytes, qh[8], scales[8]
        const char * b = row + sb*sizeof(block_iq2_s);
        r.d = *(const uint16_t *) b;
        r.a = moe_iq_u16x2(b + 2 + 4*ib);
        r.b = moe_iq_u16x2(b + 2 + 32 + 4*ib);
        r.c = (uint32_t) (uint8_t) b[2 + 64 + ib] | ((uint32_t) (uint8_t) b[2 + 64 + 8 + ib] << 8);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        // 110 bytes per 256: d, qs[64] (8 grid bytes per sub-block), qh[8], signs[32], scales[4]
        const char * b = row + sb*sizeof(block_iq3_s);
        r.d = *(const uint16_t *) b;
        r.a = moe_iq_u16x2(b + 2 + 8*ib);
        r.e = moe_iq_u16x2(b + 2 + 8*ib + 4);
        r.b = moe_iq_u16x2(b + 2 + 64 + 8 + 4*ib);
        r.c = (uint32_t) (uint8_t) b[2 + 64 + ib] | ((uint32_t) (uint8_t) b[2 + 64 + 8 + 32 + ib/2] << 8);
    } else {
        static_assert(type == GGML_TYPE_IQ3_XXS, "unsupported type");
        // 98 bytes: d, qs[64] (grid indices, 4 values each), then per sub-block 4 x 7-bit sign sets | scale << 28
        const char * b = row + sb*sizeof(block_iq3_xxs);
        r.d = *(const uint16_t *) b;
        r.a = moe_iq_u16x2(b + 2 + 8*ib);
        r.b = moe_iq_u16x2(b + 6 + 8*ib);
        r.c = moe_iq_u16x2(b + 2 + QK_K/4 + 4*ib);
    }
    return r;
}

// The kernels are bound by LDS and VALU work per item (gfx906 counters, gate at 1 token: ~76 VALU, ~14 LDS and ~3.4
// VMEM instructions per 32 values), not by memory; computing the sign masks in registers instead of the LDS table
// measured no gain (tg 61.3 -> 60.9 t/s).
// SR: the sign byte masks computed in registers instead of read from the sg table (the same bytes, so the same sums):
// 4 sign bits spread to the byte LSBs by one 24-bit multiply (bits land at 0, 8, 16, 24 without carries), times 255 for
// the 0xFF masks. 7-bit sign sets get bit 7 = their parity (as ksigns_iq2xs). The flat kernel is bound by its random LDS
// gathers (grid + signs), so trading 8 sign gathers per gate/up pair for VALU can pay there (GGML_CUDA_MOE_IQ_FLAT_SREG).
static __device__ __forceinline__ uint2 moe_sign_mask(const uint32_t s8, uint2 & t) {
    t.x = __umul24(s8 & 0xF, 0x204081u) & 0x01010101u;
    t.y = __umul24((s8 >> 4) & 0xF, 0x204081u) & 0x01010101u;
    return make_uint2((t.x << 8) - t.x, (t.y << 8) - t.y);
}
static __device__ __forceinline__ uint32_t moe_sign7(const uint32_t s7) {
    return s7 | ((__builtin_popcount(s7) & 1u) << 7);
}

template <ggml_type type, bool SR = false>
static __device__ __forceinline__ float moe_iq_dot(const moe_iq_raw & r, const block_q8_1 * __restrict__ ys, const int it,
        const uint2 * __restrict__ g2, const uint32_t * __restrict__ g3, const uint2 * __restrict__ sg) {
    const block_q8_1 * yb = ys + it;
    const int * yq = (const int *) yb->qs;
    const float d = __half2float(__ushort_as_half(r.d));
    if constexpr (type == GGML_TYPE_IQ2_XS) {
        int s[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q = ((k < 2 ? r.a : r.b) >> (16*(k & 1))) & 0xFFFF;
            const uint2 g = g2[q & 0x1FF];
            int v0, v1;
            if constexpr (SR) {
                uint2 t;
                const uint2 m = moe_sign_mask(moe_sign7(q >> 9), t);
                v0 = (int) ((g.x ^ m.x) + t.x);
                v1 = (int) ((g.y ^ m.y) + t.y);
            } else {
                const uint2 m = sg[q >> 9];
                v0 = (int) ((g.x ^ m.x) + (m.x & 0x01010101u));
                v1 = (int) ((g.y ^ m.y) + (m.y & 0x01010101u));
            }
            s[k] = ggml_cuda_dp4a(v1, yq[2*k + 1], ggml_cuda_dp4a(v0, yq[2*k], 0));
        }
        const int si = (2*(int) (r.c & 0xF) + 1)*(s[0] + s[1]) + (2*(int) (r.c >> 4) + 1)*(s[2] + s[3]);
        return d*0.125f*__low2float(yb->ds)*(float) si;
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        // non-linear 4-bit values from kvalues_iq4nl by byte permutes (no tables in LDS)
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int2 v = get_int_from_table_16((int) (k == 0 ? r.a : k == 1 ? r.b : k == 2 ? r.c : r.e), kvalues_iq4nl);
            s = ggml_cuda_dp4a(v.y, yq[k + 4], ggml_cuda_dp4a(v.x, yq[k], s));
        }
        return d*__low2float(yb->ds)*(float) (r.s*s);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        // 8 values per grid entry (iq2xxs_grid in g2), signs as for iq2_xs; d*(0.5 + ls)/4 = d*(2*ls + 1)/8
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint2 g = g2[(r.a >> (8*k)) & 0xFF];
            int v0, v1;
            if constexpr (SR) {
                uint2 t;
                const uint2 m = moe_sign_mask(moe_sign7((r.b >> (7*k)) & 0x7F), t);
                v0 = (int) ((g.x ^ m.x) + t.x);
                v1 = (int) ((g.y ^ m.y) + t.y);
            } else {
                const uint2 m = sg[(r.b >> (7*k)) & 0x7F];
                v0 = (int) ((g.x ^ m.x) + (m.x & 0x01010101u));
                v1 = (int) ((g.y ^ m.y) + (m.y & 0x01010101u));
            }
            s = ggml_cuda_dp4a(v1, yq[2*k + 1], ggml_cuda_dp4a(v0, yq[2*k], s));
        }
        return d*0.125f*__low2float(yb->ds)*(float) ((2*(int) (r.b >> 28) + 1)*s);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t ii = k < 2 ? r.a : r.b;
            const uint32_t gl = g3[(ii >> (16*(k & 1)))     & 0xFF];
            const uint32_t gh = g3[(ii >> (16*(k & 1) + 8)) & 0xFF];
            int v0, v1;
            if constexpr (SR) {
                uint2 t;
                const uint2 m = moe_sign_mask(moe_sign7((r.c >> (7*k)) & 0x7F), t);
                v0 = (int) ((gl ^ m.x) + t.x);
                v1 = (int) ((gh ^ m.y) + t.y);
            } else {
                const uint2 m = sg[(r.c >> (7*k)) & 0x7F];
                v0 = (int) ((gl ^ m.x) + (m.x & 0x01010101u));
                v1 = (int) ((gh ^ m.y) + (m.y & 0x01010101u));
            }
            s = ggml_cuda_dp4a(v1, yq[2*k + 1], ggml_cuda_dp4a(v0, yq[2*k], s));
        }
        return d*0.25f*__low2float(yb->ds)*(float) ((2*(int) (r.c >> 28) + 1)*s);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        // non-linear 4-bit values from kvalues_iq4nl by byte permutes (no tables in LDS), as the iq4_xs case
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int2 v = get_int_from_table_16((int) (k == 0 ? r.a : k == 1 ? r.b : k == 2 ? r.c : r.e), kvalues_iq4nl);
            s = ggml_cuda_dp4a(v.y, yq[k + 4], ggml_cuda_dp4a(v.x, yq[k], s));
        }
        return d*__low2float(yb->ds)*(float) s;
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        // 2-bit crumbs unpacked by the HIP perm path of vec_dot_q2_0_q8_1 (sign-extended 16-bit words)
        int s = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q  = (int) (int16_t) (uint16_t) ((j < 2 ? r.a : r.b) >> (16*(j & 1)));
            const int u  = get_int_b4(yb->qs, 2*j + 0);
            const int v  = get_int_b4(yb->qs, 2*j + 1);
            const uint32_t qx_indices = ((uint32_t) q & 0x03) | (((uint32_t) q & 0x0C) << 6) | (((uint32_t) q & 0x30) << 12) | (((uint32_t) q & 0xC0) << 18);
            const uint32_t qy_bits    = (uint32_t) q >> 8;
            const uint32_t qy_indices = (qy_bits & 0x03) | ((qy_bits & 0x0C) << 6) | ((qy_bits & 0x30) << 12) | ((qy_bits & 0xC0) << 18);
            const int qx = __builtin_amdgcn_perm(0x020100FF, 0x020100FF, qx_indices);
            const int qy = __builtin_amdgcn_perm(0x020100FF, 0x020100FF, qy_indices);
            s = ggml_cuda_dp4a(u, qx, s);
            s = ggml_cuda_dp4a(v, qy, s);
        }
        return d*__low2float(yb->ds)*(float) s;
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        // 10-bit grid index (8-bit qs + 2 qh bits), direct sign bits, TWO_SC nibble scales:
        // the arithmetic of vec_dot_iq2_s_q8_1 (sumi0/sumi1 = the two 16-value halves)
        const uint8_t * qs = (const uint8_t *) &r.a;
        const uint8_t * signs_packed_8 = (const uint8_t *) &r.b;
        const int qh  = (int) (r.c & 0xFF);
        const int ls0 = (int) ((r.c >> 8) & 0x0F);
        const int ls1 = (int) ((r.c >> 8) >> 4);
        int sumi0 = 0;
        int sumi1 = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 g = g2[qs[l0/2] | ((qh << (8 - l0)) & 0x300)];
            // sign byte -> byte masks; -v = (v ^ 0xFF) + 1 per byte (grid values are never 0, so no carry), as __vsub4
            int grid_l, grid_h;
            if constexpr (SR) {
                uint2 t;
                const uint2 m = moe_sign_mask(signs_packed_8[l0/2], t);
                grid_l = (int) ((g.x ^ m.x) + t.x);
                grid_h = (int) ((g.y ^ m.y) + t.y);
            } else {
                const uint2 m = sg[signs_packed_8[l0/2]];
                grid_l = (int) ((g.x ^ m.x) + (m.x & 0x01010101u));
                grid_h = (int) ((g.y ^ m.y) + (m.y & 0x01010101u));
            }
            const int u0 = get_int_b4(yb->qs, l0 + 0);
            const int u1 = get_int_b4(yb->qs, l0 + 1);
            if (l0 < 4) {
                sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
                sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
            } else {
                sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
                sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
            }
        }
        const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
        return d*__low2float(yb->ds)*(float) sumi;
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        // 9-bit grid index (8-bit qs + 1 qh bit), direct sign bits, parity-selected scale nibble:
        // the arithmetic of vec_dot_iq3_s_q8_1
        const int2 qp = make_int2((int) r.a, (int) r.e);
        const uint8_t * qs = (const uint8_t *) &qp;
        const uint8_t * signs_packed_8 = (const uint8_t *) &r.b;
        const int qh = (int) (r.c & 0xFF);
        int sumi = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t gl = g3[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
            const uint32_t gh = g3[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
            int grid_l, grid_h;
            if constexpr (SR) {
                uint2 t;
                const uint2 m = moe_sign_mask(signs_packed_8[l0/2], t);
                grid_l = (int) ((gl ^ m.x) + t.x);
                grid_h = (int) ((gh ^ m.y) + t.y);
            } else {
                const uint2 m = sg[signs_packed_8[l0/2]]; // as iq2_s
                grid_l = (int) ((gl ^ m.x) + (m.x & 0x01010101u));
                grid_h = (int) ((gh ^ m.y) + (m.y & 0x01010101u));
            }
            const int u0 = get_int_b4(yb->qs, l0 + 0);
            const int u1 = get_int_b4(yb->qs, l0 + 1);
            sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
        }
        sumi *= 1 + 2*((int) (((r.c >> 8) >> (4*(it & 1))) & 0x0F));
        return d*__low2float(yb->ds)*(float) sumi;
    } else {
        static_assert(type == GGML_TYPE_IQ3_XXS, "unsupported type");
    }
}

// GLU: gate (vg) and up (vu) of one type, dst = SwiGLU(-clamp) of the pair. A block stages the grid / sign tables and
// the activation row in LDS (dynamic part: ncols_x/QK8_1 blocks of block_q8_1) and walks RG row groups of 256/LPR rows,
// so that setup and its barrier are paid once per RG groups: with one group per block it was the cost of the down
// projection's many small blocks (TP4, K 512: 8192 blocks at 4 tokens, 142 us). The tables stay in LDS: gfx906 serves
// the lanes' scattered grid lookups from global memory (L1) far slower (gate 21 -> 32 us at 1 token). Each lane keeps
// PF items in flight (the next PF load while the current ones are summed).
#ifndef MOE_IQ_PF
#define MOE_IQ_PF 2
#endif
template <ggml_type type, bool GLU, int LPR, int RG>
__launch_bounds__(256)
static __global__ void moe_coal_iq(
        const char * __restrict__ vg, const char * __restrict__ vu, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t sg_row, const int64_t sg_exp, const int64_t su_row, const int64_t su_exp,
        const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst, const int64_t s2_dst, const moe_direct_t direct) {
    constexpr bool X2 = type == GGML_TYPE_IQ2_XS;
    constexpr bool I4 = type == GGML_TYPE_IQ4_XS; // no LDS tables
    constexpr bool XX = type == GGML_TYPE_IQ2_XXS;
    constexpr bool NL = type == GGML_TYPE_IQ4_NL; // no LDS tables (kvalues permutes)
    constexpr bool Q20 = type == GGML_TYPE_Q2_0; // no LDS tables
    constexpr bool I2S = type == GGML_TYPE_IQ2_S; // iq2s_grid in g2, direct signs
    constexpr bool I3S = type == GGML_TYPE_IQ3_S; // iq3s_grid in g3, direct signs
    constexpr int  PF = MOE_IQ_PF;
    __shared__ uint2    g2[X2 ? 512 : XX ? 256 : I2S ? 1024 : 1];  // iq2xs_grid / iq2xxs_grid / iq2s_grid
    __shared__ uint32_t g3[X2 || I4 || XX || NL || Q20 || I2S ? 1 : I3S ? 512 : 256]; // iq3xxs_grid / iq3s_grid
    // ksigns_iq2xs expanded to byte masks (0x00 / 0xFF per value); iq2_s / iq3_s: every sign byte (direct sign bits)
    __shared__ uint2    sg[I4 || NL || Q20 ? 1 : I2S || I3S ? 256 : 128];
    extern __shared__ int4 ys_raw[];
    block_q8_1 * ys = (block_q8_1 *) ys_raw;

    const int slot   = blockIdx.y;
    const int expert = direct.ids[(slot / n_expert_used)*direct.s1 + slot % n_expert_used];
    const int nyb    = ncols_x / QK8_1;
    const int sub    = threadIdx.x % LPR;
    if constexpr (X2) {
        for (int i = threadIdx.x; i < 512; i += 256) {
            const uint64_t g = iq2xs_grid[i];
            g2[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
        }
    } else if constexpr (XX) {
        const uint64_t g = iq2xxs_grid[threadIdx.x];
        g2[threadIdx.x] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
    } else if constexpr (I2S) {
        for (int i = threadIdx.x; i < 1024; i += 256) {
            const uint64_t g = iq2s_grid[i];
            g2[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
        }
    } else if constexpr (!I4 && !NL && !Q20) {
        if constexpr (I3S) {
            for (int i = threadIdx.x; i < 512; i += 256) {
                g3[i] = iq3s_grid[i];
            }
        } else {
            g3[threadIdx.x] = iq3xxs_grid[threadIdx.x];
        }
    }
    if ((I2S || I3S || (!I4 && !NL && !Q20 && threadIdx.x < 128))) {
        const uint32_t s = I2S || I3S ? threadIdx.x : ksigns_iq2xs[threadIdx.x];
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            lo |= ((s >> b)       & 1u) ? (0xFFu << (8*b)) : 0u;
            hi |= ((s >> (b + 4)) & 1u) ? (0xFFu << (8*b)) : 0u;
        }
        sg[threadIdx.x] = make_uint2(lo, hi);
    }
    {
        const int * src = (const int *) (vy + (int64_t) (slot / direct.y_div)*stride_col_y);
        int * d = (int *) ys;
        for (int i = threadIdx.x; i < nyb*(int) (sizeof(block_q8_1)/4); i += 256) {
            d[i] = src[i];
        }
    }
    __syncthreads();

#pragma unroll 1
    for (int g = 0; g < RG; ++g) {
        const int row = (blockIdx.x*RG + g)*(256/LPR) + threadIdx.x / LPR;
        if (row >= nrows_x) {
            break; // whole lane groups; nothing after the loop needs the block
        }
        const char * rg = vg + expert*sg_exp + row*sg_row;
        const char * ru = GLU ? vu + expert*su_exp + row*su_row : nullptr;

        moe_iq_raw rgb[PF], rub[PF];
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int it = sub + p*LPR;
            if (it < nyb) {
                rgb[p] = moe_iq_load<type>(rg, it);
                if constexpr (GLU) {
                    rub[p] = moe_iq_load<type>(ru, it);
                }
            }
        }
        float ag = 0.0f, au = 0.0f;
        for (int it0 = sub; it0 < nyb; it0 += PF*LPR) {
            moe_iq_raw ngb[PF], nub[PF];
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                const int it = it0 + (PF + p)*LPR;
                if (it < nyb) {
                    ngb[p] = moe_iq_load<type>(rg, it);
                    if constexpr (GLU) {
                        nub[p] = moe_iq_load<type>(ru, it);
                    }
                }
            }
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                const int it = it0 + p*LPR;
                if (it < nyb) {
                    ag += moe_iq_dot<type>(rgb[p], ys, it, g2, g3, sg);
                    if constexpr (GLU) {
                        au += moe_iq_dot<type>(rub[p], ys, it, g2, g3, sg);
                    }
                }
            }
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                rgb[p] = ngb[p];
                if constexpr (GLU) {
                    rub[p] = nub[p];
                }
            }
        }
#pragma unroll
        for (int off = LPR/2; off > 0; off >>= 1) {
            ag += __shfl_xor(ag, off, LPR);
            if constexpr (GLU) {
                au += __shfl_xor(au, off, LPR);
            }
        }
        if (sub == 0) {
            float * o = dst + (slot / n_expert_used)*s2_dst + (slot % n_expert_used)*s1_dst + row;
            *o = GLU ? moe_glu(ag, au, direct.glu_limit) : ag;
        }
    }
}

// One token (every slot reads the same activation row): moe_coal_iq gave each (row group, slot) its own block, so the
// table + activation staging was paid by every block and a decode MUL_MAT_ID (6..8 slots of 512 rows) ran ~100 blocks:
// 1.5 waves per SIMD, latency bound (gfx906: the per-item work is ~76 VALU + ~14 LDS for 32 values). Here a grid of a
// few blocks per CU stages once and walks (slot, row group) tasks with a block stride; LPR lanes per row (16 / 32: short
// per-lane chains). GGML_CUDA_MOE_IQ_FLAT=0 off, GGML_CUDA_MOE_IQ_FLAT_LPR, GGML_CUDA_MOE_IQ_FLAT_BPC (blocks per CU).
template <ggml_type type, bool GLU, int LPR, int PF, bool SR = false>
__launch_bounds__(256)
static __global__ void moe_coal_iq_flat(
        const char * __restrict__ vg, const char * __restrict__ vu, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t sg_row, const int64_t sg_exp, const int64_t su_row, const int64_t su_exp,
        const int n_expert_used, const int64_t s1_dst, const moe_direct_t direct, const int n_tok, const int64_t s2_dst,
        const int64_t stride_col_y) {
    constexpr bool X2 = type == GGML_TYPE_IQ2_XS;
    constexpr bool I4 = type == GGML_TYPE_IQ4_XS;
    constexpr bool XX = type == GGML_TYPE_IQ2_XXS;
    constexpr bool NL = type == GGML_TYPE_IQ4_NL;
    constexpr bool Q20 = type == GGML_TYPE_Q2_0;
    constexpr bool I2S = type == GGML_TYPE_IQ2_S;
    constexpr bool I3S = type == GGML_TYPE_IQ3_S;
    constexpr int  RPT = 256/LPR; // rows per task
    __shared__ uint2    g2[X2 ? 512 : XX ? 256 : I2S ? 1024 : 1];
    __shared__ uint32_t g3[X2 || I4 || XX || NL || Q20 || I2S ? 1 : I3S ? 512 : 256];
    __shared__ uint2    sg[I4 || NL || Q20 ? 1 : I2S || I3S ? 256 : 128];
    extern __shared__ int4 ys_raw[];
    block_q8_1 * ys = (block_q8_1 *) ys_raw;

    const int nyb = ncols_x / QK8_1;
    const int sub = threadIdx.x % LPR;
    if constexpr (X2) {
        for (int i = threadIdx.x; i < 512; i += 256) {
            const uint64_t g = iq2xs_grid[i];
            g2[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
        }
    } else if constexpr (XX) {
        const uint64_t g = iq2xxs_grid[threadIdx.x];
        g2[threadIdx.x] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
    } else if constexpr (I2S) {
        for (int i = threadIdx.x; i < 1024; i += 256) {
            const uint64_t g = iq2s_grid[i];
            g2[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
        }
    } else if constexpr (!I4 && !NL && !Q20) {
        if constexpr (I3S) {
            for (int i = threadIdx.x; i < 512; i += 256) {
                g3[i] = iq3s_grid[i];
            }
        } else {
            g3[threadIdx.x] = iq3xxs_grid[threadIdx.x];
        }
    }
    if (!SR && (I2S || I3S || (!I4 && !NL && !Q20 && threadIdx.x < 128))) {
        const uint32_t sv = I2S || I3S ? threadIdx.x : ksigns_iq2xs[threadIdx.x];
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            lo |= ((sv >> b)       & 1u) ? (0xFFu << (8*b)) : 0u;
            hi |= ((sv >> (b + 4)) & 1u) ? (0xFFu << (8*b)) : 0u;
        }
        sg[threadIdx.x] = make_uint2(lo, hi);
    }
    // the tokens' activation rows, nyb blocks each, back to back
    for (int t = 0; t < n_tok; ++t) {
        const int * src = (const int *) (vy + (int64_t) t*stride_col_y);
        int * d = (int *) (ys + t*nyb);
        for (int i = threadIdx.x; i < nyb*(int) (sizeof(block_q8_1)/4); i += 256) {
            d[i] = src[i];
        }
    }
    __syncthreads();

    const int n_rg    = (nrows_x + RPT - 1)/RPT;
    const int n_tasks = n_rg*n_expert_used*n_tok;
#pragma unroll 1
    for (int task = blockIdx.x; task < n_tasks; task += gridDim.x) {
        const int slot   = task / n_rg;          // token*n_expert_used + k
        const int tok    = slot / n_expert_used;
        const int kk     = slot % n_expert_used;
        const int row    = (task % n_rg)*RPT + threadIdx.x / LPR;
        const int expert = direct.ids[tok*direct.s1 + kk];
        const block_q8_1 * yt = ys + tok*nyb;
        if (row >= nrows_x) {
            continue; // whole lane groups; the loop has no barrier
        }
        const char * rg = vg + expert*sg_exp + row*sg_row;
        const char * ru = GLU ? vu + expert*su_exp + row*su_row : nullptr;

        moe_iq_raw rgb[PF], rub[PF];
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int it = sub + p*LPR;
            if (it < nyb) {
                rgb[p] = moe_iq_load<type>(rg, it);
                if constexpr (GLU) {
                    rub[p] = moe_iq_load<type>(ru, it);
                }
            }
        }
        float ag = 0.0f, au = 0.0f;
        for (int it0 = sub; it0 < nyb; it0 += PF*LPR) {
            moe_iq_raw ngb[PF], nub[PF];
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                const int it = it0 + (PF + p)*LPR;
                if (it < nyb) {
                    ngb[p] = moe_iq_load<type>(rg, it);
                    if constexpr (GLU) {
                        nub[p] = moe_iq_load<type>(ru, it);
                    }
                }
            }
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                const int it = it0 + p*LPR;
                if (it < nyb) {
                    ag += moe_iq_dot<type, SR>(rgb[p], yt, it, g2, g3, sg);
                    if constexpr (GLU) {
                        au += moe_iq_dot<type, SR>(rub[p], yt, it, g2, g3, sg);
                    }
                }
            }
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                rgb[p] = ngb[p];
                if constexpr (GLU) {
                    rub[p] = nub[p];
                }
            }
        }
#pragma unroll
        for (int off = LPR/2; off > 0; off >>= 1) {
            ag += __shfl_xor(ag, off, LPR);
            if constexpr (GLU) {
                au += __shfl_xor(au, off, LPR);
            }
        }
        if (sub == 0) {
            float * o = dst + tok*s2_dst + kk*s1_dst + row;
            *o = GLU ? moe_glu(ag, au, direct.glu_limit) : ag;
        }
    }
}

// ---- launchers (the dispatch moe-vec.cu had inline) ----

template <ggml_type type, bool GLU>
static void moe_iq_flat_go(const moe_iq_args & a, const int lpr, const int pf, const bool sreg) {
    auto go = [&](auto kern, int l) {
        const int64_t n_tasks = ((a.ne01 + 256/l - 1)/(256/l))*a.n_used*a.ne12;
        const int grid = (int) std::min<int64_t>(n_tasks, (int64_t) a.max_grid);
        kern<<<grid, 256, a.shm, a.stream>>>(a.vg, a.vu, a.vy, a.dst, (int) a.ne00, (int) a.ne01, a.sg_row, a.sg_exp,
            a.su_row, a.su_exp, (int) a.n_used, a.s1_dst, a.dr, (int) a.ne12, a.s2_dst, a.s_col);
    };
#define MOE_IQ_FLAT_GO(l) if (sreg && pf == 1) { go(moe_coal_iq_flat<type, GLU, l, 1, true>, l); } else \
    switch (pf) { case 4: go(moe_coal_iq_flat<type, GLU, l, 4>, l); break; \
    case 3: go(moe_coal_iq_flat<type, GLU, l, 3>, l); break; case 1: go(moe_coal_iq_flat<type, GLU, l, 1>, l); break; \
    default: go(moe_coal_iq_flat<type, GLU, l, 2>, l); break; }
    switch (lpr) {
        case 8:  MOE_IQ_FLAT_GO(8);  break;
        case 32: MOE_IQ_FLAT_GO(32); break;
        default: MOE_IQ_FLAT_GO(16); break;
    }
#undef MOE_IQ_FLAT_GO
}

template <ggml_type type, bool GLU>
static void moe_iq_coal_go(const moe_iq_args & a, const int lpr, const int rg) {
    auto go = [&](auto kern, int l, int r) {
        const int rows_pb = (256/l)*r;
        kern<<<dim3((a.ne01 + rows_pb - 1)/rows_pb, a.n_slots), 256, a.shm, a.stream>>>(a.vg, a.vu, a.vy, a.dst,
            (int) a.ne00, (int) a.ne01, a.sg_row, a.sg_exp, a.su_row, a.su_exp, a.s_col, (int) a.n_used, a.s1_dst, a.s2_dst, a.dr);
    };
#define MOE_IQ_RG(l) switch (rg) { case 2: go(moe_coal_iq<type, GLU, l, 2>, l, 2); break; \
    case 4: go(moe_coal_iq<type, GLU, l, 4>, l, 4); break; default: go(moe_coal_iq<type, GLU, l, 1>, l, 1); break; }
    switch (lpr) {
        case 16: MOE_IQ_RG(16) break;
        case 32: MOE_IQ_RG(32) break;
        default: MOE_IQ_RG(8)  break;
    }
#undef MOE_IQ_RG
}

// iq4_xs / iq4_nl / q2_0 have no gate/up pair variant (as before: their single-projection kernel even when glu is set)
template <ggml_type type> static constexpr bool moe_iq_has_glu() {
    return type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_IQ4_NL && type != GGML_TYPE_Q2_0;
}

template <ggml_type type>
void moe_iq_flat_launch(const moe_iq_args & a, const bool glu, const int lpr, const int pf, const bool sreg) {
    if constexpr (moe_iq_has_glu<type>()) {
        if (glu) {
            moe_iq_flat_go<type, true>(a, lpr, pf, sreg);
            return;
        }
    }
    moe_iq_flat_go<type, false>(a, lpr, pf, sreg);
}

template <ggml_type type>
void moe_iq_coal_launch(const moe_iq_args & a, const bool glu, const int lpr, const int rg) {
    if constexpr (moe_iq_has_glu<type>()) {
        if (glu) {
            moe_iq_coal_go<type, true>(a, lpr, rg);
            return;
        }
    }
    moe_iq_coal_go<type, false>(a, lpr, rg);
}

#define MOE_IQ_INSTANCE(t) \
    template void moe_iq_flat_launch<t>(const moe_iq_args &, bool, int, int, bool); \
    template void moe_iq_coal_launch<t>(const moe_iq_args &, bool, int, int);
