#include "gcn-q6k-dec.cuh"
#include "mmvq.cuh"
#include "vecdotq.cuh"

// q6_K x q8_1 matvec at one token on GCN. An item is 32 values of a super-block (two 16-value runs A and B, 64 values
// apart, from 16 bytes of ql and 16 of qh); LPR lanes of a row take consecutive items so a wave streams whole
// super-blocks. Each lane loads the raw words of all its items first (13 loads per item), then the block stages the
// activations in LDS (int8 runs, q8_1 scales, 16-run sums for the -32 offset) while those loads are in flight.
// GGML_CUDA_Q6K_DEC=0: off (MMVQ); GGML_CUDA_Q6K_DEC_LPR: lanes per row.

static __device__ __forceinline__ void q6d_join16(const int * w, const int sh, int * out) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = (int) __builtin_amdgcn_alignbyte((uint32_t) w[i + 1], (uint32_t) w[i], sh);
    }
}

typedef int q6d_i4 __attribute__((ext_vector_type(4), aligned(4)));
typedef int q6d_i2 __attribute__((ext_vector_type(2), aligned(4)));

// weights of one item of one row, loaded ahead of the math
struct q6d_raw {
    int ql[5], qh[5]; // aligned-down dwords around the 16 bytes of ql / qh (2-byte aligned data)
    int sc[2];        // aligned-down dwords holding scales 8n+c and 8n+c+4
    int d;            // super-block scale (half bits)
};

// activation words of one item (runs A and B) and their q8_1 scales (half2 bits)
struct q6d_act {
    int ya[4], yb[4];
    int dsa, dsb;
};

// sum over the LPR lanes of a group (lane 0 of the group holds it): DPP inside 16 lanes, readlane across rows of 16
template <int LPR>
static __device__ __forceinline__ float q6d_group_sum(float x) {
#define Q6D_DPP(v_, ctl_) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v_), ctl_, 0xF, 0xF, false))
    x += Q6D_DPP(x, 0xB1);
    x += Q6D_DPP(x, 0x4E);
    if (LPR == 8) {
        x += Q6D_DPP(x, 0x141); // row_half_mirror
    } else {
        x += Q6D_DPP(x, 0x124); // row_ror:4
        x += Q6D_DPP(x, 0x128); // row_ror:8
    }
#undef Q6D_DPP
    if (LPR > 16) {
        const int xi = __float_as_int(x);
        const float r0 = __int_as_float(__builtin_amdgcn_readlane(xi, 0)), r1 = __int_as_float(__builtin_amdgcn_readlane(xi, 16));
        if (LPR == 32) {
            const float r2 = __int_as_float(__builtin_amdgcn_readlane(xi, 32)), r3 = __int_as_float(__builtin_amdgcn_readlane(xi, 48));
            return (threadIdx.x & 32) ? r2 + r3 : r0 + r1;
        }
        const float r2 = __int_as_float(__builtin_amdgcn_readlane(xi, 32)), r3 = __int_as_float(__builtin_amdgcn_readlane(xi, 48));
        return (r0 + r1) + (r2 + r3);
    }
    return x;
}

// 32-bit offset from a uniform base: lets the compiler use the scalar-base + vgpr-offset load form
#define Q6D_AT(base, off) ((base) + (uint64_t) (uint32_t) (off))

// LPR lanes take the items of a row (item index lane + j*LPR), R rows per lane group share the activation loads and sums;
// GUARD: 8*nb is not a multiple of LPR (the last round reads item 0 again and adds nothing)
template <int LPR, int NI, int R, int BT, bool GUARD>
__launch_bounds__(BT)
static __global__ void gcn_q6k_dec(const char * __restrict__ W, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int K, const int M, const uint32_t w_stride) {
    const int nit = (K / QK_K)*8;
    const int lane = threadIdx.x % LPR;
    constexpr int RPW = (64/LPR)*R; // rows per wave
    const int wave_row0 = __builtin_amdgcn_readfirstlane((blockIdx.x*(BT/64) + threadIdx.x/64)*RPW);
    const int wr0 = wave_row0 < M ? wave_row0 : M - 1;
    const int grp = (threadIdx.x % 64)/LPR;
    const char * wb = W + (int64_t) wr0*w_stride;
    const char * yb = (const char *) y;

    const int q = lane & 7, n = q >> 2, c = q & 3;
    const int shl = c < 2 ? 0 : 2;
    uint32_t woff[R];
#pragma unroll
    for (int k = 0; k < R; ++k) {
        const int r = wave_row0 + grp*R + k;
        woff[k] = (uint32_t) ((r < M ? r : M - 1) - wr0)*w_stride + (lane >> 3)*(uint32_t) sizeof(block_q6_K);
    }
    const uint32_t wstep = (LPR/8)*(uint32_t) sizeof(block_q6_K);
    const uint32_t oql = 64*n + 16*c, oqh = 128 + 32*n + 16*(c & 1), osc = 192 + 8*n + c;
    const uint32_t ra0 = (lane >> 3)*16 + 8*n + c;
    const uint32_t yoff_a = (ra0 >> 1)*sizeof(block_q8_1), yoff_b = yoff_a + 2*sizeof(block_q8_1);
    const uint32_t ysub = 4 + 16*(c & 1);
    const uint32_t ystep = (LPR/8)*8*sizeof(block_q8_1);
    const uint32_t wbase = (uint32_t) (uintptr_t) wb;

    q6d_act act[NI];
    q6d_raw raw[R][NI];
#pragma unroll
    for (int j = 0; j < NI; ++j) {
        const bool ok = !GUARD || lane + j*LPR < nit;
        const uint32_t jy = ok ? j*ystep : 0;
        const q6d_i4 va = *(const q6d_i4 *) Q6D_AT(yb, yoff_a + ysub + jy);
        const q6d_i4 vb = *(const q6d_i4 *) Q6D_AT(yb, yoff_b + ysub + jy);
        act[j].ya[0] = va.x; act[j].ya[1] = va.y; act[j].ya[2] = va.z; act[j].ya[3] = va.w;
        act[j].yb[0] = vb.x; act[j].yb[1] = vb.y; act[j].yb[2] = vb.z; act[j].yb[3] = vb.w;
        act[j].dsa = *(const int *) Q6D_AT(yb, yoff_a + jy);
        act[j].dsb = *(const int *) Q6D_AT(yb, yoff_b + jy);
#pragma unroll
        for (int k = 0; k < R; ++k) {
            const uint32_t o = woff[k] + (ok ? j*wstep : 0);
            uint32_t o1 = o + oql, o2 = o + oqh, o3 = o + osc;
            const uint32_t x1 = o1 + wbase, x2 = o2 + wbase;
            o1 -= x1 & 3; o2 -= x2 & 3; o3 -= (wbase + o3) & 3;
            const q6d_i4 v1 = *(const q6d_i4 *) Q6D_AT(wb, o1);
            const q6d_i4 v2 = *(const q6d_i4 *) Q6D_AT(wb, o2);
            raw[k][j].ql[0] = v1.x; raw[k][j].ql[1] = v1.y; raw[k][j].ql[2] = v1.z; raw[k][j].ql[3] = v1.w;
            raw[k][j].qh[0] = v2.x; raw[k][j].qh[1] = v2.y; raw[k][j].qh[2] = v2.z; raw[k][j].qh[3] = v2.w;
            raw[k][j].ql[4] = *(const int *) Q6D_AT(wb, o1 + ((x1 & 3) ? 16 : 12));
            raw[k][j].qh[4] = *(const int *) Q6D_AT(wb, o2 + ((x2 & 3) ? 16 : 12));
            const q6d_i2 sv = *(const q6d_i2 *) Q6D_AT(wb, o3);
            raw[k][j].sc[0] = sv.x; raw[k][j].sc[1] = sv.y;
            raw[k][j].d = *(const unsigned short *) Q6D_AT(wb, o + 208);
        }
    }

    __builtin_amdgcn_sched_barrier(0); // keep every load ahead of the math (the scheduler would put them next to their uses)

    float acc[R];
#pragma unroll
    for (int k = 0; k < R; ++k) {
        acc[k] = 0.0f;
    }
#pragma unroll
    for (int j = 0; j < NI; ++j) {
        const bool ok = !GUARD || lane + j*LPR < nit;
        int sa = 0, sb = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sa = ggml_cuda_dp4a(act[j].ya[i], 0x01010101, sa);
            sb = ggml_cuda_dp4a(act[j].yb[i], 0x01010101, sb);
        }
        const float fa = __low2float(*(const half2 *) &act[j].dsa);
        const float fb = __low2float(*(const half2 *) &act[j].dsb);
#pragma unroll
        for (int k = 0; k < R; ++k) {
            const uint32_t o = woff[k] + (ok ? j*wstep : 0);
            const int shq = (int) ((wbase + o + oql) & 3);
            const int shg = (int) ((wbase + o + oqh) & 3);
            const int shs = (int) ((wbase + o + osc) & 3);
            int ql[4], qh[4];
            q6d_join16(raw[k][j].ql, shq, ql);
            q6d_join16(raw[k][j].qh, shg, qh);
            int da = 0, db = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int qa = (ql[i] & 0x0F0F0F0F) | ((qh[i] << (4 - shl)) & 0x30303030);
                const int qb = ((ql[i] >> 4) & 0x0F0F0F0F) | ((qh[i] >> shl) & 0x30303030);
                da = ggml_cuda_dp4a(qa, act[j].ya[i], da);
                db = ggml_cuda_dp4a(qb, act[j].yb[i], db);
            }
            const uint64_t sw = (uint64_t) (uint32_t) raw[k][j].sc[0] | ((uint64_t) (uint32_t) raw[k][j].sc[1] << 32);
            const int ia = (int) (int8_t) (sw >> (8*shs))*(da - 32*sa);
            const int ib = (int) (int8_t) (sw >> (8*shs + 32))*(db - 32*sb);
            const float d = __half2float(__ushort_as_half((unsigned short) raw[k][j].d));
            acc[k] += ok ? d*(fa*(float) ia + fb*(float) ib) : 0.0f;
        }
    }
#pragma unroll
    for (int k = 0; k < R; ++k) {
        const float v = q6d_group_sum<LPR>(acc[k]);
        const int r = wave_row0 + grp*R + k;
        if (lane == 0 && r < M) {
            dst[r] = v;
        }
    }
}

// 2..4 tokens (MTP verify): one row per lane group, the weights of a lane's items loaded once (ahead of the math) and
// decoded once, then each token's dot. Per token the sums are formed exactly as in gcn_q6k_dec (same lanes, items, order
// and group sum; the row count per group does not enter a row's sum), so column t equals the 1-token result for column t.
template <int LPR, int NI, int NT, bool GUARD>
__launch_bounds__(64)
static __global__ void gcn_q6k_dec_nt(const char * __restrict__ W, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int K, const int M, const uint32_t w_stride, const uint32_t y_stride, const int d_stride) {
    const int nit = (K / QK_K)*8;
    const int lane = threadIdx.x % LPR;
    constexpr int RPW = 64/LPR; // rows per wave (= block)
    const int wave_row0 = __builtin_amdgcn_readfirstlane(blockIdx.x*RPW);
    const int wr0 = wave_row0 < M ? wave_row0 : M - 1;
    const int grp = threadIdx.x/LPR;
    const char * wb = W + (int64_t) wr0*w_stride;
    const char * yb = (const char *) y;

    const int q = lane & 7, n = q >> 2, c = q & 3;
    const int shl = c < 2 ? 0 : 2;
    const int r = wave_row0 + grp;
    const uint32_t woff = (uint32_t) ((r < M ? r : M - 1) - wr0)*w_stride + (lane >> 3)*(uint32_t) sizeof(block_q6_K);
    const uint32_t wstep = (LPR/8)*(uint32_t) sizeof(block_q6_K);
    const uint32_t oql = 64*n + 16*c, oqh = 128 + 32*n + 16*(c & 1), osc = 192 + 8*n + c;
    const uint32_t ra0 = (lane >> 3)*16 + 8*n + c;
    const uint32_t yoff_a = (ra0 >> 1)*sizeof(block_q8_1), yoff_b = yoff_a + 2*sizeof(block_q8_1);
    const uint32_t ysub = 4 + 16*(c & 1);
    const uint32_t ystep = (LPR/8)*8*sizeof(block_q8_1);
    const uint32_t wbase = (uint32_t) (uintptr_t) wb;

    q6d_raw raw[NI];
#pragma unroll
    for (int j = 0; j < NI; ++j) {
        const bool ok = !GUARD || lane + j*LPR < nit;
        const uint32_t o = woff + (ok ? j*wstep : 0);
        uint32_t o1 = o + oql, o2 = o + oqh, o3 = o + osc;
        const uint32_t x1 = o1 + wbase, x2 = o2 + wbase;
        o1 -= x1 & 3; o2 -= x2 & 3; o3 -= (wbase + o3) & 3;
        const q6d_i4 v1 = *(const q6d_i4 *) Q6D_AT(wb, o1);
        const q6d_i4 v2 = *(const q6d_i4 *) Q6D_AT(wb, o2);
        raw[j].ql[0] = v1.x; raw[j].ql[1] = v1.y; raw[j].ql[2] = v1.z; raw[j].ql[3] = v1.w;
        raw[j].qh[0] = v2.x; raw[j].qh[1] = v2.y; raw[j].qh[2] = v2.z; raw[j].qh[3] = v2.w;
        raw[j].ql[4] = *(const int *) Q6D_AT(wb, o1 + ((x1 & 3) ? 16 : 12));
        raw[j].qh[4] = *(const int *) Q6D_AT(wb, o2 + ((x2 & 3) ? 16 : 12));
        const q6d_i2 sv = *(const q6d_i2 *) Q6D_AT(wb, o3);
        raw[j].sc[0] = sv.x; raw[j].sc[1] = sv.y;
        raw[j].d = *(const unsigned short *) Q6D_AT(wb, o + 208);
    }

    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        acc[t] = 0.0f;
    }
#pragma unroll
    for (int j = 0; j < NI; ++j) {
        const bool ok = !GUARD || lane + j*LPR < nit;
        const uint32_t jy = ok ? j*ystep : 0;
        const uint32_t o = woff + (ok ? j*wstep : 0);
        const int shq = (int) ((wbase + o + oql) & 3);
        const int shg = (int) ((wbase + o + oqh) & 3);
        const int shs = (int) ((wbase + o + osc) & 3);
        int ql[4], qh[4];
        q6d_join16(raw[j].ql, shq, ql);
        q6d_join16(raw[j].qh, shg, qh);
        int qa[4], qb[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            qa[i] = (ql[i] & 0x0F0F0F0F) | ((qh[i] << (4 - shl)) & 0x30303030);
            qb[i] = ((ql[i] >> 4) & 0x0F0F0F0F) | ((qh[i] >> shl) & 0x30303030);
        }
        const uint64_t sw = (uint64_t) (uint32_t) raw[j].sc[0] | ((uint64_t) (uint32_t) raw[j].sc[1] << 32);
        const int sca = (int) (int8_t) (sw >> (8*shs));
        const int scb = (int) (int8_t) (sw >> (8*shs + 32));
        const float d = __half2float(__ushort_as_half((unsigned short) raw[j].d));
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const uint32_t yt = t*y_stride + jy;
            const q6d_i4 va = *(const q6d_i4 *) Q6D_AT(yb, yoff_a + ysub + yt);
            const q6d_i4 vb = *(const q6d_i4 *) Q6D_AT(yb, yoff_b + ysub + yt);
            const int dsa = *(const int *) Q6D_AT(yb, yoff_a + yt);
            const int dsb = *(const int *) Q6D_AT(yb, yoff_b + yt);
            const int ya[4] = {va.x, va.y, va.z, va.w}, ybv[4] = {vb.x, vb.y, vb.z, vb.w};
            int sa = 0, sb = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sa = ggml_cuda_dp4a(ya[i], 0x01010101, sa);
                sb = ggml_cuda_dp4a(ybv[i], 0x01010101, sb);
            }
            const float fa = __low2float(*(const half2 *) &dsa);
            const float fb = __low2float(*(const half2 *) &dsb);
            int da = 0, db = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                da = ggml_cuda_dp4a(qa[i], ya[i], da);
                db = ggml_cuda_dp4a(qb[i], ybv[i], db);
            }
            const int ia = sca*(da - 32*sa);
            const int ib = scb*(db - 32*sb);
            acc[t] += ok ? d*(fa*(float) ia + fb*(float) ib) : 0.0f;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float v = q6d_group_sum<LPR>(acc[t]);
        if (lane == 0 && r < M) {
            dst[(int64_t) t*d_stride + r] = v;
        }
    }
}

// lanes per row: the fewest (>= 8) that leave <= 3 items per lane (measured: more lanes beat more items in flight per lane)
static int q6d_pick_lpr(const int nit) {
    int lpr = 8;
    while (lpr < 64 && (nit + lpr - 1)/lpr > 3) {
        lpr *= 2;
    }
    return lpr;
}

bool ggml_cuda_gcn_q6k_dec_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_Q6K_DEC"); return !e || atoi(e) != 0; }();
    // GGML_CUDA_Q6K_DEC_NT: 2..4 tokens too (gcn_q6k_dec_nt, the MTP verify); 0 = MMVQ there
    static const bool nt_env = [] { const char * e = getenv("GGML_CUDA_Q6K_DEC_NT"); return e && atoi(e) != 0; }();
    const int64_t K = src0->ne[0];
    return env && GGML_CUDA_CC_IS_GCN(cc) && src0->type == GGML_TYPE_Q6_K && K % QK_K == 0 && K <= 16384 &&
        src0->ne[1] <= INT_MAX && src0->ne[2] == 1 && src0->ne[3] == 1 && ggml_is_contiguous(src0) &&
        src0->nb[1] % 2 == 0 && ((uintptr_t) src0->data) % 2 == 0 &&
        src1->type == GGML_TYPE_F32 && (src1->ne[1] == 1 || (nt_env && src1->ne[1] <= 4 && K <= 8192)) && src1->ne[2] == 1 &&
        src1->ne[3] == 1 && src1->nb[0] == sizeof(float) && dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float) &&
        dst->nb[1] % sizeof(float) == 0;
}

void ggml_cuda_gcn_q6k_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const int lpr_env = [] { const char * e = getenv("GGML_CUDA_Q6K_DEC_LPR"); return e ? atoi(e) : 0; }();
    const int64_t K = src0->ne[0], M = src0->ne[1];
    const int64_t K_pad = GGML_PAD(K, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> yq_local(ctx.pool());
    const block_q8_1 * yq = (const block_q8_1 *) ggml_cuda_q8_1_activation(ctx, src1, src0->type, K_pad, yq_local);
    const int nit = (int) (K / QK_K)*8;
    const int lpr = lpr_env ? lpr_env : q6d_pick_lpr(nit);
    const int ni = (nit + lpr - 1)/lpr;
    const bool guard = nit % lpr != 0;
    const int64_t N = src1->ne[1];
    if (N > 1) {
        const uint32_t y_stride = (uint32_t) ((K_pad/QK8_1)*sizeof(block_q8_1));
        const int d_stride = (int) (dst->nb[1]/sizeof(float));
        const dim3 grid_nt((unsigned) ((M + 64/lpr - 1)/(64/lpr)));
        cudaStream_t st = ctx.stream();
#define Q6DNT1(L_, N_, T_, G_) gcn_q6k_dec_nt<L_, N_, T_, G_><<<grid_nt, 64, 0, st>>>((const char *) src0->data, yq, \
        (float *) dst->data, (int) K, (int) M, (uint32_t) src0->nb[1], y_stride, d_stride)
#define Q6DNT_T(L_, N_, G_) switch (N) { case 2: Q6DNT1(L_, N_, 2, G_); break; case 3: Q6DNT1(L_, N_, 3, G_); break; \
        default: Q6DNT1(L_, N_, 4, G_); break; }
#define Q6DNT(L_, N_) if (guard) { Q6DNT_T(L_, N_, true) } else { Q6DNT_T(L_, N_, false) }
#define Q6DNT_N(L_) switch (ni) { case 1: Q6DNT(L_, 1) break; case 2: Q6DNT(L_, 2) break; case 3: Q6DNT(L_, 3) break; \
        case 4: Q6DNT(L_, 4) break; default: GGML_ABORT("q6k_dec_nt: items per lane"); }
        switch (lpr) {
            case 8:  Q6DNT_N(8)  break;
            case 16: Q6DNT_N(16) break;
            case 32: Q6DNT_N(32) break;
            default: Q6DNT_N(64) break;
        }
#undef Q6DNT_N
#undef Q6DNT
#undef Q6DNT_T
#undef Q6DNT1
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    static const int r_env = [] { const char * e = getenv("GGML_CUDA_Q6K_DEC_R"); return e ? atoi(e) : 0; }();
    const int rr = r_env ? r_env : (nit >= 128 && M >= 2048) ? 2 : 1;
    const int bt = 64;
    const int rpw = (64/lpr)*rr;
    const dim3 grid((unsigned) ((M + rpw - 1)/rpw));
    cudaStream_t stream = ctx.stream();
#define Q6D1(L_, N_, R_, G_) gcn_q6k_dec<L_, N_, R_, 64, G_><<<grid, bt, 0, stream>>>((const char *) src0->data, yq, \
        (float *) dst->data, (int) K, (int) M, (uint32_t) src0->nb[1])
#define Q6D(L_, N_) switch (rr*2 + guard) { case 2: Q6D1(L_, N_, 1, false); break; case 3: Q6D1(L_, N_, 1, true); break; \
        case 4: Q6D1(L_, N_, 2, false); break; case 5: Q6D1(L_, N_, 2, true); break; default: GGML_ABORT("q6k_dec: rows"); }
#define Q6D_N(L_) switch (ni) { case 1: Q6D(L_, 1) break; case 2: Q6D(L_, 2) break; case 3: Q6D(L_, 3) break; \
        case 4: Q6D(L_, 4) break; case 5: Q6D(L_, 5) break; case 6: Q6D(L_, 6) break; case 7: Q6D(L_, 7) break; \
        case 8: Q6D(L_, 8) break; case 9: Q6D(L_, 9) break; case 10: Q6D(L_, 10) break; case 12: Q6D(L_, 12) break; default: GGML_ABORT("q6k_dec: items per lane"); }
    switch (lpr) {
        case 8:  Q6D_N(8)  break;
        case 16: Q6D_N(16) break;
        case 32: Q6D_N(32) break;
        default: Q6D_N(64) break;
    }
#undef Q6D_N
#undef Q6D
#undef Q6D1
    CUDA_CHECK(cudaGetLastError());
}
