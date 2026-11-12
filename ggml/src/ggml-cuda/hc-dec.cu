#include "hc-dec.cuh"
#include "mmvq.cuh"

#include <cstdlib>

// Decode-sized hyper-connection kernels for GCN (few tokens). At batch 1 the generic path spends most of the time
// on launch overhead and short-K matrix-vector products that leave most lanes idle; these kernels fuse the chains
// and give every lane one weight row.

static int hc_dec_env(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

// 16 bytes of streamed weights: a non-temporal load on HIP, so the weights do not push the shared activations out of
// the caches (gfx906 HC matvecs: 11.7 -> 9.8 us up, 12.5 -> 11.1 us down)
static __device__ __forceinline__ uint4 hc_dec_ld_stream(const void * p) {
#if defined(GGML_USE_HIP)
    typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
    const u32x4 t = __builtin_nontemporal_load((const u32x4 *) p);
    return make_uint4(t.x, t.y, t.z, t.w);
#else
    return *(const uint4 *) p;
#endif
}

// SCALE -> SILU -> q8_0 up projection -> RESHAPE -> gated DSV4_HC_PRE:
//   a[k,t]     = silu(s_lo*lo[k,t])
//   g[c,e,t]   = sum_k W[c*E + e, k] * a[k,t]
//   pre[e,t]   = s_pre * sum_c xn[e,c,t] * sigmoid(g[c,e,t])
// Block = 64 output columns x HC_DEC_NS streams (one wave per stream, one lane per weight row); the activations sit in
// LDS (broadcast reads), the weights are read as raw q8_0 (pairs of blocks = 17 dwords) and multiplied in float, so
// no activation quantization kernel is needed. The gate is never stored.
#define HC_DEC_NS 4
#ifndef HC_DEC_WIDE
#define HC_DEC_WIDE 0
#endif
#ifndef HC_DEC_TUNROLL
#define HC_DEC_TUNROLL 4
#endif
#ifndef HC_DEC_PRELOAD
#define HC_DEC_PRELOAD 5
#endif

// T lanes share a weight row and split its block pairs (lane = row*T + sub: the T lanes of a row read adjacent pairs),
// so a block covers 64/T output columns and the grid has T times more blocks than one lane per row would give
template <int NT, int T>
__launch_bounds__(64*HC_DEC_NS)
static __global__ void hc_up_pre_dec(
        const char * __restrict__ W, const float * __restrict__ lo, const float * __restrict__ xn, float * __restrict__ dst,
        const int K, const int64_t w_stride, const int E, const float s_lo, const float s_pre,
        const int64_t lo_st, const int64_t xn_sc, const int64_t xn_st, const int64_t d_st) {
    constexpr int CPB = 64/T; // output columns per block
    extern __shared__ float hc_act[]; // [NT][K]
    __shared__ float red[HC_DEC_NS][NT][CPB];

    const int tid = threadIdx.x;
    for (int i = tid; i < NT*K; i += 64*HC_DEC_NS) {
        const int t = i / K;
        const int k = i - t*K;
        const float v = s_lo*lo[t*lo_st + k];
        hc_act[i] = v/(1.0f + expf(-v));
    }
    __syncthreads();

    const int  c     = tid / 64;
    const int  lane  = tid % 64;
    const int  r     = lane / T;
    const int  sub   = lane % T;
    const int  e     = blockIdx.x*CPB + r;
    const bool valid = e < E;
    const int * wr   = (const int *) (W + (int64_t) (c*E + (valid ? e : E - 1))*w_stride);

    float acc[NT] = {0.0f};
    const int npairs = K / 64;
    // one lane per row (T == 1) with K = 320: all 5 pairs are loaded before any is used, so the lane waits for one
    // memory round trip instead of five (few waves per CU: the loop was bound by the per-pair load latency)
    if (T == 1 && NT == 1 && npairs == HC_DEC_PRELOAD) {
        int wall[HC_DEC_PRELOAD][17];
#pragma unroll
        for (int p = 0; p < HC_DEC_PRELOAD; ++p) {
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                wall[p][i] = wr[17*p + i];
            }
        }
#pragma unroll
        for (int p = 0; p < HC_DEC_PRELOAD; ++p) {
            const int * w = wall[p];
            const float d0 = __half2float(__ushort_as_half((unsigned short) (w[0] & 0xFFFF)));
            const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[8] >> 16)));
            int q0[8], q1[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                q0[j] = __builtin_amdgcn_alignbyte(w[j + 1], w[j], 2);
                q1[j] = w[9 + j];
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4 * a = (const float4 *) (hc_act + t*K + 64*p);
                float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float4 a0 = a[j];
                    const float4 a1 = a[8 + j];
                    s0 += (float) (int8_t) (q0[j]      )*a0.x + (float) (int8_t) (q0[j] >>  8)*a0.y +
                          (float) (int8_t) (q0[j] >> 16)*a0.z + (float) (int8_t) (q0[j] >> 24)*a0.w;
                    s1 += (float) (int8_t) (q1[j]      )*a1.x + (float) (int8_t) (q1[j] >>  8)*a1.y +
                          (float) (int8_t) (q1[j] >> 16)*a1.z + (float) (int8_t) (q1[j] >> 24)*a1.w;
                }
                acc[t] += d0*s0 + d1*s1;
            }
        }
    } else
    for (int p = sub; p < npairs; p += T) {
        int w[17];
        if (HC_DEC_WIDE) {
            // 68 bytes as 4 x 16-byte + 1 x 4-byte loads (dword-aligned, gfx9 unaligned mode): each load instruction
            // touches 64 cache lines (one row per lane), so fewer, wider loads cut the L1/TA line requests ~3x
            const int4 * w4 = (const int4 *) (wr + 17*p);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int4 v = w4[i];
                w[4*i + 0] = v.x; w[4*i + 1] = v.y; w[4*i + 2] = v.z; w[4*i + 3] = v.w;
            }
            w[16] = wr[17*p + 16];
        } else {
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                w[i] = wr[17*p + i];
            }
        }
        const float d0 = __half2float(__ushort_as_half((unsigned short) (w[0] & 0xFFFF)));
        const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[8] >> 16)));
        int q0[8], q1[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            q0[j] = __builtin_amdgcn_alignbyte(w[j + 1], w[j], 2); // quants of block 0 start at byte 2
            q1[j] = w[9 + j];                                     // block 1: d at bytes 34-35, quants from byte 36
        }
        // tokens fully unrolled (131-147 VGPRs, 1 wave per SIMD at 4 tokens, but limiting the unroll was slower)
#pragma unroll HC_DEC_TUNROLL
        for (int t = 0; t < NT; ++t) {
            const float4 * a = (const float4 *) (hc_act + t*K + 64*p);
            float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 a0 = a[j];
                const float4 a1 = a[8 + j];
                s0 += (float) (int8_t) (q0[j]      )*a0.x + (float) (int8_t) (q0[j] >>  8)*a0.y +
                      (float) (int8_t) (q0[j] >> 16)*a0.z + (float) (int8_t) (q0[j] >> 24)*a0.w;
                s1 += (float) (int8_t) (q1[j]      )*a1.x + (float) (int8_t) (q1[j] >>  8)*a1.y +
                      (float) (int8_t) (q1[j] >> 16)*a1.z + (float) (int8_t) (q1[j] >> 24)*a1.w;
            }
            acc[t] += d0*s0 + d1*s1;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int off = T/2; off > 0; off >>= 1) {
            acc[t] += __shfl_xor(acc[t], off, 64);
        }
    }

    if (sub == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            red[c][t][r] = valid ? xn[e + c*xn_sc + t*xn_st] / (1.0f + expf(-acc[t])) : 0.0f;
        }
    }
    __syncthreads();
    if (c == 0 && sub == 0 && valid) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float s = 0.0f;
#pragma unroll
            for (int cc = 0; cc < HC_DEC_NS; ++cc) {
                s += red[cc][t][r];
            }
            dst[e + t*d_st] = s_pre*s;
        }
    }
}

// bf16 weights: HC_DEC_BF16_L lanes per row, each lane streams its NP 16-byte chunks (8 weights each) before any is
// used, so the lane waits for one memory round trip; 64/L rows per wave, the HC_DEC_NS waves of a block are the streams.
// gfx906, E = 2560, K = 320: 8 lanes x 5 chunks 11.7 us, 4 x 10 non-temporal 8.9 us (a plain read of the bytes 9.2)
#define HC_DEC_BF16_L  4
#define HC_DEC_BF16_NP 10
#define HC_UP_R1       3

template <int NT, int NP>
__launch_bounds__(64*HC_DEC_NS)
static __global__ void hc_up_pre_dec_bf16(
        const char * __restrict__ W, const float * __restrict__ lo, const float * __restrict__ xn, float * __restrict__ dst,
        const int K, const int64_t w_stride, const int E, const float s_lo, const float s_pre,
        const int64_t lo_st, const int64_t xn_sc, const int64_t xn_st, const int64_t d_st) {
    constexpr int L  = HC_DEC_BF16_L;
    constexpr int RW = 64/L; // rows per wave
    extern __shared__ float hc_act[]; // [NT][K]
    __shared__ float red[HC_DEC_NS][NT][RW];

    const int  tid   = threadIdx.x;
    const int  c     = tid / 64;
    const int  lane  = tid % 64;
    const int  r     = lane / L;
    const int  sub   = lane % L;
    const int  e     = blockIdx.x*RW + r;
    const bool valid = e < E;
    const char * wr  = W + (int64_t) (c*E + (valid ? e : E - 1))*w_stride;

    // the weight loads go out first: the activation prologue runs while they are in flight
    uint4 w[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        w[p] = hc_dec_ld_stream(wr + 16*(sub + L*p));
    }

    for (int i = tid; i < NT*K; i += 64*HC_DEC_NS) {
        const int t = i / K;
        const int k = i - t*K;
        const float v = s_lo*lo[t*lo_st + k];
        hc_act[i] = v/(1.0f + expf(-v));
    }
    __syncthreads();

    float acc[NT] = {0.0f};
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        const uint32_t q[4] = { w[p].x, w[p].y, w[p].z, w[p].w };
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 * a = (const float4 *) (hc_act + t*K + 8*(sub + L*p));
            const float4 a0 = a[0];
            const float4 a1 = a[1];
            float s = acc[t];
            s = fmaf(__uint_as_float(q[0] << 16),          a0.x, s);
            s = fmaf(__uint_as_float(q[0] & 0xFFFF0000u),  a0.y, s);
            s = fmaf(__uint_as_float(q[1] << 16),          a0.z, s);
            s = fmaf(__uint_as_float(q[1] & 0xFFFF0000u),  a0.w, s);
            s = fmaf(__uint_as_float(q[2] << 16),          a1.x, s);
            s = fmaf(__uint_as_float(q[2] & 0xFFFF0000u),  a1.y, s);
            s = fmaf(__uint_as_float(q[3] << 16),          a1.z, s);
            s = fmaf(__uint_as_float(q[3] & 0xFFFF0000u),  a1.w, s);
            acc[t] = s;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int off = L/2; off > 0; off >>= 1) {
            acc[t] += __shfl_xor(acc[t], off, 64);
        }
    }

    if (sub == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            red[c][t][r] = valid ? xn[e + c*xn_sc + t*xn_st] / (1.0f + expf(-acc[t])) : 0.0f;
        }
    }
    __syncthreads();
    if (c == 0 && sub == 0 && valid) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float s = 0.0f;
#pragma unroll
            for (int cc = 0; cc < HC_DEC_NS; ++cc) {
                s += red[cc][t][r];
            }
            dst[e + t*d_st] = s_pre*s;
        }
    }
}

// one token with the q8_1 copy of the output for the projections that read it (q8 cache, no separate quantize launch):
// a 512-thread block takes 32 rows, two groups of the 4 stream waves above (same per-row work and arithmetic), so a
// q8_1 block lies inside one block and is quantized from LDS as quantize_q8_1 does (same lanes, same reductions)
template <int NP, int NT = 1>
__launch_bounds__(128*HC_DEC_NS)
static __global__ void hc_up_pre_dec_bf16_q8(
        const char * __restrict__ W, const float * __restrict__ lo, const float * __restrict__ xn, float * __restrict__ dst,
        block_q8_1 * __restrict__ yq, const int K, const int64_t w_stride, const int E, const float s_lo, const float s_pre,
        const int64_t xn_sc, const int64_t lo_st, const int64_t xn_st, const int64_t d_st) {
    constexpr int L  = HC_DEC_BF16_L;
    constexpr int RW = 64/L; // rows per wave
    extern __shared__ float hc_act[]; // [NT][K]
    __shared__ float red[2][HC_DEC_NS][NT][RW];
    __shared__ float outv[NT][2*RW];

    const int  tid   = threadIdx.x;
    const int  g     = tid / (64*HC_DEC_NS);  // row group
    const int  c     = (tid / 64) % HC_DEC_NS; // stream
    const int  lane  = tid % 64;
    const int  r     = lane / L;
    const int  sub   = lane % L;
    const int  e     = blockIdx.x*(2*RW) + g*RW + r;
    const bool valid = e < E;
    const char * wr  = W + (int64_t) (c*E + (valid ? e : E - 1))*w_stride;

    // HC_UP_R1 chunks before the block sync, the rest after: gfx906 streams faster with fewer bytes in flight per CU
    // (in the model 13.3 -> 12.2 us; 5 + 5: 12.5, 7 + 3: 12.8)
    uint4 w[NP];
#pragma unroll
    for (int p = 0; p < HC_UP_R1; ++p) {
        w[p] = hc_dec_ld_stream(wr + 16*(sub + L*p));
    }

    for (int i = tid; i < NT*K; i += 128*HC_DEC_NS) {
        const int t = i / K;
        const int k = i - t*K;
        const float v = s_lo*lo[t*lo_st + k];
        hc_act[i] = v/(1.0f + expf(-v));
    }
    __syncthreads();
#pragma unroll
    for (int p = HC_UP_R1; p < NP; ++p) {
        w[p] = hc_dec_ld_stream(wr + 16*(sub + L*p));
    }

    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        acc[t] = 0.0f;
    }
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        const uint32_t q[4] = { w[p].x, w[p].y, w[p].z, w[p].w };
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 * a = (const float4 *) (hc_act + t*K + 8*(sub + L*p));
            const float4 a0 = a[0];
            const float4 a1 = a[1];
            float s = acc[t];
            s = fmaf(__uint_as_float(q[0] << 16),          a0.x, s);
            s = fmaf(__uint_as_float(q[0] & 0xFFFF0000u),  a0.y, s);
            s = fmaf(__uint_as_float(q[1] << 16),          a0.z, s);
            s = fmaf(__uint_as_float(q[1] & 0xFFFF0000u),  a0.w, s);
            s = fmaf(__uint_as_float(q[2] << 16),          a1.x, s);
            s = fmaf(__uint_as_float(q[2] & 0xFFFF0000u),  a1.y, s);
            s = fmaf(__uint_as_float(q[3] << 16),          a1.z, s);
            s = fmaf(__uint_as_float(q[3] & 0xFFFF0000u),  a1.w, s);
            acc[t] = s;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int off = L/2; off > 0; off >>= 1) {
            acc[t] += __shfl_xor(acc[t], off, 64);
        }
    }
    if (sub == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            red[g][c][t][r] = valid ? xn[e + c*xn_sc + t*xn_st] / (1.0f + expf(-acc[t])) : 0.0f;
        }
    }
    __syncthreads();
    if (c == 0 && sub == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float s = 0.0f;
#pragma unroll
            for (int cc = 0; cc < HC_DEC_NS; ++cc) {
                s += red[g][cc][t][r];
            }
            const float v = s_pre*s;
            outv[t][g*RW + r] = valid ? v : 0.0f;
            if (valid) {
                dst[e + t*d_st] = v;
            }
        }
    }
    __syncthreads();
    // token t's q8_1 block of these 32 rows: 32 lanes, as quantize_q8_1
    if (tid < QK8_1*NT) {
        const int   t    = tid / QK8_1;
        const int   l    = tid % QK8_1;
        const float xi   = outv[t][l];
        const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
        const float sum  = warp_reduce_sum<QK8_1>(xi);
        const float d    = amax / 127.0f;
        block_q8_1 * b = yq + (int64_t) t*(E/QK8_1) + blockIdx.x;
        b->qs[l] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
        if (l == 0) {
            b->ds = make_half2(d, sum);
        }
    }
}

bool ggml_cuda_hc_up_pre_dec_supported(int cc, const ggml_tensor * scale, const ggml_tensor * silu, const ggml_tensor * mm,
                                       const ggml_tensor * pre) {
    static const int enabled = hc_dec_env("GGML_CUDA_HC_DEC", 1);
    static const int bf16_ok = hc_dec_env("GGML_CUDA_HC_DEC_BF16", 1);
    if (!enabled || !GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }
    const ggml_tensor * lo = scale->src[0];
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * xn = pre->src[0];
    const int64_t nt = lo->ne[1];
    return ggml_get_op_params_f32(scale, 1) == 0.0f && // no bias
        lo->type == GGML_TYPE_F32 && lo->ne[2] == 1 && lo->ne[3] == 1 && lo->nb[0] == sizeof(float) &&
        nt >= 1 && nt <= 4 && mm->src[1] == silu &&
        (w->type == GGML_TYPE_Q8_0 || (w->type == GGML_TYPE_BF16 && bf16_ok && w->ne[0] == 8*HC_DEC_BF16_L*HC_DEC_BF16_NP &&
                                       w->nb[1] % 16 == 0 && ((uintptr_t) w->data) % 16 == 0)) &&
        ggml_is_contiguous(w) && w->ne[0] == lo->ne[0] && w->ne[0] % 64 == 0 &&
        w->ne[0]*nt*sizeof(float) <= 32768 && ((uintptr_t) w->data) % 4 == 0 &&
        ggml_get_op_params_i32(pre, 1) != 0 && // gated
        xn->type == GGML_TYPE_F32 && xn->ne[1] == HC_DEC_NS && xn->ne[2] == nt && xn->nb[0] == sizeof(float) &&
        w->ne[1] == HC_DEC_NS*xn->ne[0] && pre->type == GGML_TYPE_F32 && pre->nb[0] == sizeof(float) &&
        pre->ne[0] == xn->ne[0] && pre->ne[1] == nt;
}

// one token: a q8 cache slot for the q8_1 copy of the HC_PRE output, for the projections that read it (no separate
// quantize launch; GGML_CUDA_HC_UP_Q8=0 off). Needs the type-independent cache (the default). nullptr: none
static block_q8_1 * hc_up_q8_slot(ggml_backend_cuda_context & ctx, const ggml_tensor * pre) {
    static const bool q8_env = [] {
        const char * e = getenv("GGML_CUDA_HC_UP_Q8");      const bool on  = !e || atoi(e) != 0;
        const char * c = getenv("GGML_CUDA_Q8_CACHE");      const bool qc  = !c || atoi(c) != 0;
        const char * a = getenv("GGML_CUDA_Q8_CACHE_ANYTYPE"); const bool any = !a || atoi(a) != 0;
        return on && qc && any;
    }();
    // GGML_CUDA_HC_UP_Q8_NT=0: one token only (2..4 tokens, the MTP verify, then quantize separately)
    static const bool nt_env = [] { const char * e = getenv("GGML_CUDA_HC_UP_Q8_NT"); return !e || atoi(e) != 0; }();
    const int64_t E  = pre->ne[0];
    const int64_t nt = pre->ne[1];
    if (!q8_env || E % MATRIX_ROW_PADDING != 0 || !ggml_is_contiguous(pre) || nt < 1 || nt > 4 || (nt > 1 && !nt_env) ||
            pre->ne[2] != 1 || pre->ne[3] != 1) {
        return nullptr;
    }
    const ggml_tensor * key = ggml_cuda_q8_cache_key(pre);
    for (const auto & ent : ctx.q8_cache) {
        if (ent.t == key && ent.ne[0] == E && ent.ne[1] == nt && ent.ne[2] == 1 && ent.ne[3] == 1) {
            return nullptr;
        }
    }
    ctx.q8_cache.push_back({ key, (int) GGML_TYPE_Q8_0, { E, nt, 1, 1 },
                             std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), (size_t) nt*E/QK8_1*sizeof(block_q8_1)) });
    return (block_q8_1 *) ctx.q8_cache.back().buf->get();
}

void ggml_cuda_hc_up_pre_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, const ggml_tensor * mm, ggml_tensor * pre) {
    const ggml_tensor * lo = scale->src[0];
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * xn = pre->src[0];
    const int K  = (int) w->ne[0];
    const int E  = (int) xn->ne[0];
    const int nt = (int) lo->ne[1];
    static const int t_env = hc_dec_env("GGML_CUDA_HC_DEC_T", 0);
    // lanes per weight row: gfx906, E = 2560, K = 320: 8 at one token (more blocks), 1 at 2-4 tokens (verify step
    // 89.0 -> 93.5 t/s against 4). Tried and slower or equal: dp4a on int8-quantized activations, weights staged
    // through LDS with coalesced loads (the kernel is bound by its chain of dependent loads, not ALU or access pattern)
    // one token: 1 lane per row with all 5 block pairs preloaded (one memory round trip): 14.0 -> 10.6 us per call
    // (8 lanes per row before). Several tokens: 2 lanes per row sharing the tokens, and fewer unrolled tokens, both
    // measured no better (16.0 us at 4 tokens).
    const int T = t_env ? t_env : 1;
    const dim3 grid((E + 64/T - 1)/(64/T));
    const size_t smem = (size_t) nt*K*sizeof(float);
    const float s_lo  = ggml_get_op_params_f32(scale, 0);
    const float s_pre = ggml_get_op_params_f32(pre, 0);
    cudaStream_t stream = ctx.stream();
    if (w->type == GGML_TYPE_BF16) {
        const dim3 grid_b((E + 64/HC_DEC_BF16_L - 1)/(64/HC_DEC_BF16_L));
        block_q8_1 * yq = hc_up_q8_slot(ctx, pre);
#define HC_UP_PRE_BF16(NT) hc_up_pre_dec_bf16<NT, HC_DEC_BF16_NP><<<grid_b, 64*HC_DEC_NS, smem, stream>>>((const char *) w->data, \
        (const float *) lo->data, (const float *) xn->data, (float *) pre->data, K, w->nb[1], E, s_lo, s_pre, lo->nb[1]/sizeof(float), \
        xn->nb[1]/sizeof(float), xn->nb[2]/sizeof(float), pre->nb[1]/sizeof(float))
        if (yq != nullptr) {
#define HC_UP_Q8(NT) hc_up_pre_dec_bf16_q8<HC_DEC_BF16_NP, NT><<<E/(2*64/HC_DEC_BF16_L), 128*HC_DEC_NS, smem, stream>>>( \
                (const char *) w->data, (const float *) lo->data, (const float *) xn->data, (float *) pre->data, yq, K, w->nb[1], E, \
                s_lo, s_pre, xn->nb[1]/sizeof(float), lo->nb[1]/sizeof(float), xn->nb[2]/sizeof(float), pre->nb[1]/sizeof(float))
            switch (nt) {
                case 1:  HC_UP_Q8(1); break;
                case 2:  HC_UP_Q8(2); break;
                case 3:  HC_UP_Q8(3); break;
                default: HC_UP_Q8(4); break;
            }
#undef HC_UP_Q8
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        switch (nt) {
            case 1:  HC_UP_PRE_BF16(1); break;
            case 2:  HC_UP_PRE_BF16(2); break;
            case 3:  HC_UP_PRE_BF16(3); break;
            default: HC_UP_PRE_BF16(4); break;
        }
#undef HC_UP_PRE_BF16
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#define HC_UP_PRE(NT, T_) hc_up_pre_dec<NT, T_><<<grid, 64*HC_DEC_NS, smem, stream>>>((const char *) w->data, (const float *) lo->data, \
        (const float *) xn->data, (float *) pre->data, K, w->nb[1], E, s_lo, s_pre, lo->nb[1]/sizeof(float), xn->nb[1]/sizeof(float), \
        xn->nb[2]/sizeof(float), pre->nb[1]/sizeof(float))
#define HC_UP_PRE_T(NT) if (T == 1) { HC_UP_PRE(NT, 1); } else if (T == 2) { HC_UP_PRE(NT, 2); } else if (T == 4) { HC_UP_PRE(NT, 4); } \
        else { HC_UP_PRE(NT, 8); }
    switch (nt) {
        case 1:  HC_UP_PRE_T(1); break;
        case 2:  HC_UP_PRE_T(2); break;
        case 3:  HC_UP_PRE_T(3); break;
        default: HC_UP_PRE_T(4); break;
    }
#undef HC_UP_PRE_T
#undef HC_UP_PRE
    CUDA_CHECK(cudaGetLastError());
}

// bf16 matvecs sharing x, one token: a block of 4 waves takes one row, each wave a quarter of K; every lane holds its
// NP 16-byte chunks and the matching x values before the first multiply (one memory round trip). The wave sums are
// added in a fixed order, so the result does not depend on scheduling. gfx906, 324 x 10240: 12.5 us (2 rows per block
// 13.9 us, 2 waves of 10 chunks 14.1, 8 waves of 8-byte chunks 41; a plain read of the same bytes 9.3).
struct bf16mv_multi_args {
    const char * w[GGML_CUDA_BF16MV_MAXM];
    float      * dst[GGML_CUDA_BF16MV_MAXM];
    int64_t      nb1[GGML_CUDA_BF16MV_MAXM];
    int          row0[GGML_CUDA_BF16MV_MAXM + 1];
    int          n;
};

#define BF16MV_NP 5
#define BF16MV_KW 4

template <int NP>
__launch_bounds__(64*BF16MV_KW)
static __global__ void bf16mv_multi_dec(const bf16mv_multi_args args, const float * __restrict__ x) {
    __shared__ float red[BF16MV_KW];

    const int wave = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int k0   = wave*(64*8*NP);

    const int g = blockIdx.x;
    int s = 0;
    while (s + 1 < args.n && g >= args.row0[s + 1]) {
        ++s;
    }
    const char * wr = args.w[s] + (g - args.row0[s])*args.nb1[s];

    uint4  w[NP];
    float4 xa[NP], xb[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        const int e = k0 + 8*(lane + 64*p);
        w[p]  = hc_dec_ld_stream(wr + 2*e);
        xa[p] = *(const float4 *) (x + e);
        xb[p] = *(const float4 *) (x + e + 4);
    }

    float acc = 0.0f;
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        const float xv[8] = { xa[p].x, xa[p].y, xa[p].z, xa[p].w, xb[p].x, xb[p].y, xb[p].z, xb[p].w };
        const uint32_t q[4] = { w[p].x, w[p].y, w[p].z, w[p].w };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc = fmaf(__uint_as_float(q[i] << 16),         xv[2*i + 0], acc);
            acc = fmaf(__uint_as_float(q[i] & 0xFFFF0000u), xv[2*i + 1], acc);
        }
    }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off, 64);
    }
    if (lane == 0) {
        red[wave] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < BF16MV_KW; ++k) {
            sum += red[k];
        }
        args.dst[s][g - args.row0[s]] = sum;
    }
}

// 2..4 tokens (MTP verify): a block takes BF16NT_R rows and every token; each thread walks K in 2048-value steps with
// 16 bytes of each row and the tokens' 8 matching values, so x is read once per BF16NT_R rows. The generic matvec
// (mul_mat_vec_f) took 22 us for the HC down + inject (324 x 10240) at 3 tokens.
#ifndef BF16NT_R
#define BF16NT_R 2 // rows per block: 2 beats 4 (MTP verify HC down 26.1 -> 23.4 us traced, 81 -> 162 blocks) and 1 (40.5: x re-read)
#endif
template <int NT>
__launch_bounds__(256)
static __global__ void bf16mv_nt_dec(const bf16mv_multi_args args, const float * __restrict__ x, const int K, const int64_t xs) {
    constexpr int R = BF16NT_R;
    const int n_rows = args.row0[args.n];
    const int tid  = threadIdx.x;
    const int lane = tid % 64, wave = tid / 64;
    const char * wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int g = min(blockIdx.x*R + r, n_rows - 1);
        int s = 0;
        while (s + 1 < args.n && g >= args.row0[s + 1]) {
            ++s;
        }
        wr[r] = args.w[s] + (g - args.row0[s])*args.nb1[s];
    }
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    for (int k0 = 8*tid; k0 < K; k0 += 8*256) {
        uint4 w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            w[r] = hc_dec_ld_stream(wr[r] + 2*k0);
        }
        float4 xa[NT], xb[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            xa[t] = *(const float4 *) (x + t*xs + k0);
            xb[t] = *(const float4 *) (x + t*xs + k0 + 4);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t q[4] = { w[r].x, w[r].y, w[r].z, w[r].w };
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float xv[8] = { xa[t].x, xa[t].y, xa[t].z, xa[t].w, xb[t].x, xb[t].y, xb[t].z, xb[t].w };
                float a = acc[r][t];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    a = fmaf(__uint_as_float(q[i] << 16),         xv[2*i + 0], a);
                    a = fmaf(__uint_as_float(q[i] & 0xFFFF0000u), xv[2*i + 1], a);
                }
                acc[r][t] = a;
            }
        }
    }
    __shared__ float red[4][R*NT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) {
                v += __shfl_xor(v, off, 64);
            }
            if (lane == 0) {
                red[wave][r*NT + t] = v;
            }
        }
    }
    __syncthreads();
    if (tid < R*NT) {
        const int r = tid / NT, t = tid % NT;
        const int g = blockIdx.x*R + r;
        if (g < n_rows) {
            int s = 0;
            while (s + 1 < args.n && g >= args.row0[s + 1]) {
                ++s;
            }
            const int rows = args.row0[s + 1] - args.row0[s];
            args.dst[s][(int64_t) t*rows + (g - args.row0[s])] = ((red[0][tid] + red[1][tid]) + red[2][tid]) + red[3][tid];
        }
    }
}

bool ggml_cuda_bf16mv_multi_supported(const ggml_tensor * mm) {
    static const int enabled = hc_dec_env("GGML_CUDA_BF16MV_MULTI", 1);
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    return enabled && mm->op == GGML_OP_MUL_MAT && w->type == GGML_TYPE_BF16 && x->type == GGML_TYPE_F32 &&
        mm->type == GGML_TYPE_F32 && w->ne[0] == BF16MV_KW*64*8*BF16MV_NP && w->ne[2] == 1 && w->ne[3] == 1 &&
        w->nb[0] == 2 && w->nb[1] % 16 == 0 && ((uintptr_t) w->data) % 16 == 0 &&
        x->ne[1] >= 1 && x->ne[1] <= 4 && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[1] % 16 == 0 &&
        x->nb[0] == sizeof(float) && ((uintptr_t) x->data) % 16 == 0 &&
        ggml_is_contiguous(mm) && w->ne[1] <= (1 << 20);
}

bool ggml_cuda_bf16mv_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * mms, int n) {
    if (n < 1 || n > GGML_CUDA_BF16MV_MAXM || !GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    bf16mv_multi_args args = {};
    args.n = n;
    args.row0[0] = 0;
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * mm = mms[k];
        if (!ggml_cuda_bf16mv_multi_supported(mm) || mm->src[1]->data != mms[0]->src[1]->data ||
                mm->src[0]->ne[0] != mms[0]->src[0]->ne[0]) {
            return false;
        }
        args.w[k]        = (const char *) mm->src[0]->data;
        args.dst[k]      = (float *) mm->data;
        args.nb1[k]      = mm->src[0]->nb[1];
        args.row0[k + 1] = args.row0[k] + (int) mm->src[0]->ne[1];
    }
    const int n_rows = args.row0[n];
    const ggml_tensor * x = mms[0]->src[1];
    const int nt = (int) x->ne[1];
    if (nt == 1) {
        bf16mv_multi_dec<BF16MV_NP><<<n_rows, 64*BF16MV_KW, 0, ctx.stream()>>>(args, (const float *) x->data);
    } else {
        const int K = (int) mms[0]->src[0]->ne[0];
        const int64_t xs = x->nb[1]/sizeof(float);
        const dim3 grid((n_rows + BF16NT_R - 1)/BF16NT_R);
        switch (nt) {
            case 2:  bf16mv_nt_dec<2><<<grid, 256, 0, ctx.stream()>>>(args, (const float *) x->data, K, xs); break;
            case 3:  bf16mv_nt_dec<3><<<grid, 256, 0, ctx.stream()>>>(args, (const float *) x->data, K, xs); break;
            default: bf16mv_nt_dec<4><<<grid, 256, 0, ctx.stream()>>>(args, (const float *) x->data, K, xs); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// 4 consecutive weights i4*4 .. i4*4 + 3 of a f32 or bf16 row
template <bool BF16>
static __device__ __forceinline__ float4 hc_dec_w4(const char * __restrict__ w, const int i4) {
    if constexpr (BF16) {
        const uint2 q = ((const uint2 *) w)[i4];
        return make_float4(__uint_as_float(q.x << 16), __uint_as_float(q.x & 0xFFFF0000u),
                           __uint_as_float(q.y << 16), __uint_as_float(q.y & 0xFFFF0000u));
    } else {
        return ((const float4 *) w)[i4];
    }
}

// GDN gate inputs at decode: MUL_MAT(alpha) -> RESHAPE -> ADD(dt) -> SOFTPLUS -> MUL(A) and MUL_MAT(beta) -> RESHAPE ->
// SIGMOID share the activation: one wave per output row (alpha rows first, then beta rows), f32 or bf16 weights
template <int NT, bool BF16>
__launch_bounds__(256)
static __global__ void gdn_ab_dec(
        const float * __restrict__ x, const char * __restrict__ wa, const char * __restrict__ wb, const float * __restrict__ dt,
        const float * __restrict__ A, float * __restrict__ gate, float * __restrict__ beta, const int K, const int H,
        const int64_t x_st, const int64_t wa_s, const int64_t wb_s, const int64_t g_st, const int64_t b_st) {
    const int r    = blockIdx.x*4 + threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    if (r >= 2*H) {
        return;
    }
    const bool is_b = r >= H;
    const int  h    = is_b ? r - H : r;
    const char * wrow = is_b ? wb + h*wb_s : wa + h*wa_s;
    float acc[NT] = {0.0f};
    for (int k = lane; k < K/4; k += 64) {
        const float4 w = hc_dec_w4<BF16>(wrow, k);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 v = ((const float4 *) (x + t*x_st))[k];
            acc[t] += w.x*v.x + w.y*v.y + w.z*v.z + w.w*v.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            acc[t] += __shfl_xor(acc[t], off, 64);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (is_b) {
                beta[h + t*b_st] = 1.0f/(1.0f + expf(-acc[t]));
            } else {
                const float v  = acc[t] + dt[h];
                const float sp = v > 20.0f ? v : logf(1.0f + expf(v)); // as ggml softplus
                gate[h + t*g_st] = sp*A[h];
            }
        }
    }
}

// nodes: mm_a, (reshape), add, softplus, mul, (reshape), mm_b, (reshape), sigmoid
bool ggml_cuda_gdn_ab_dec_supported(int cc, const ggml_tensor * mm_a, const ggml_tensor * add, const ggml_tensor * sp,
                                    const ggml_tensor * mul, const ggml_tensor * mm_b, const ggml_tensor * sig) {
    static const int enabled = hc_dec_env("GGML_CUDA_GDN_AB_DEC", 1);
    if (!enabled || !GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }
    const ggml_tensor * x  = mm_a->src[1];
    const ggml_tensor * wa = mm_a->src[0];
    const ggml_tensor * wb = mm_b->src[0];
    const ggml_tensor * dt = add->src[1];
    const ggml_tensor * A  = mul->src[0] == sp ? mul->src[1] : mul->src[0];
    const int64_t nt = x->ne[1];
    const bool w_ok = (wa->type == GGML_TYPE_F32 || wa->type == GGML_TYPE_BF16) && wb->type == wa->type &&
        wa->nb[1] % 16 == 0 && wb->nb[1] % 16 == 0;
    return mm_b->src[1] == x && x->type == GGML_TYPE_F32 && w_ok &&
        nt >= 1 && nt <= 4 && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) && x->ne[0] % 4 == 0 &&
        (x->nb[1] % 16) == 0 && ((uintptr_t) x->data) % 16 == 0 &&
        ggml_is_contiguous(wa) && ggml_is_contiguous(wb) && ggml_are_same_shape(wa, wb) && wa->ne[0] == x->ne[0] &&
        wa->ne[2] == 1 && ((uintptr_t) wa->data) % 16 == 0 && ((uintptr_t) wb->data) % 16 == 0 &&
        add->src[0]->view_src == mm_a && ggml_is_contiguous(add) && ggml_is_contiguous(mm_a) &&
        dt->type == GGML_TYPE_F32 && ggml_nelements(dt) == wa->ne[1] && ggml_is_contiguous(dt) &&
        sp->src[0] == add && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS &&
        (mul->src[0] == sp || mul->src[1] == sp) && A->type == GGML_TYPE_F32 && ggml_nelements(A) == wa->ne[1] &&
        ggml_is_contiguous(A) && ggml_is_contiguous(mul) && ggml_nelements(mul) == wa->ne[1]*nt &&
        ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID && sig->src[0]->view_src == mm_b && ggml_is_contiguous(mm_b) &&
        ggml_is_contiguous(sig) && ggml_nelements(sig) == wa->ne[1]*nt && mul->type == GGML_TYPE_F32 && sig->type == GGML_TYPE_F32;
}

void ggml_cuda_gdn_ab_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * add, const ggml_tensor * mm_b,
                          ggml_tensor * gate, ggml_tensor * sig) {
    const ggml_tensor * x  = mm_a->src[1];
    const ggml_tensor * wa = mm_a->src[0];
    const ggml_tensor * wb = mm_b->src[0];
    const ggml_tensor * sp = gate->src[0]->op == GGML_OP_UNARY ? gate->src[0] : gate->src[1];
    const ggml_tensor * A  = gate->src[0] == sp ? gate->src[1] : gate->src[0];
    const int K = (int) x->ne[0], H = (int) wa->ne[1], nt = (int) x->ne[1];
    const dim3 grid((2*H + 3)/4);
    cudaStream_t stream = ctx.stream();
#define GDN_AB(NT, BF) gdn_ab_dec<NT, BF><<<grid, 256, 0, stream>>>((const float *) x->data, (const char *) wa->data, \
        (const char *) wb->data, (const float *) add->src[1]->data, (const float *) A->data, (float *) gate->data, (float *) sig->data, \
        K, H, x->nb[1]/sizeof(float), wa->nb[1], wb->nb[1], H, H)
#define GDN_AB_NT(BF) switch (nt) { case 1: GDN_AB(1, BF); break; case 2: GDN_AB(2, BF); break; case 3: GDN_AB(3, BF); break; \
        default: GDN_AB(4, BF); break; }
    if (wa->type == GGML_TYPE_BF16) {
        GDN_AB_NT(true)
    } else {
        GDN_AB_NT(false)
    }
#undef GDN_AB_NT
#undef GDN_AB
    CUDA_CHECK(cudaGetLastError());
}

// Shared-expert gate tail: MUL_MAT(w_g [K], x) -> SIGMOID -> MUL(shexp, .) -> ADD(moe, .) in one kernel:
//   out[e,t] = moe[e,t] + shexp[e,t] * sigmoid(dot(w_g, x[:,t]))
// One block per token (the dot product needs the whole row first). Four launches -> one at decode (<= 8 tokens).
template <bool BF16>
static __global__ void __launch_bounds__(256) shexp_gate_add(
        const char * __restrict__ wg, const float * __restrict__ x, const float * __restrict__ shexp,
        const float * __restrict__ moe, float * __restrict__ out, const int K, const int E,
        const int64_t s_x, const int64_t s_sh, const int64_t s_moe, const int64_t s_out) {
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    const float4 * xp = (const float4 *) (x + t*s_x);
    float acc = 0.0f;
    for (int k = tid; k < K/4; k += 256) {
        const float4 a = xp[k], b = hc_dec_w4<BF16>(wg, k);
        acc += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    constexpr int ws = ggml_cuda_get_physical_warp_size();
    __shared__ float red[256/ws];
    acc = warp_reduce_sum<ws>(acc);
    if (tid % ws == 0) {
        red[tid / ws] = acc;
    }
    __syncthreads();
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 256/ws; ++i) {
        dot += red[i];
    }
    const float g = 1.0f / (1.0f + expf(-dot));
    for (int e = tid; e < E; e += 256) {
        out[t*s_out + e] = moe[t*s_moe + e] + shexp[t*s_sh + e]*g;
    }
}

static bool shexp_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

bool ggml_cuda_shexp_gate_add_supported(const ggml_tensor * mm, const ggml_tensor * sig, const ggml_tensor * mul,
                                        const ggml_tensor * add) {
    static const int enabled = hc_dec_env("GGML_CUDA_SHEXP_GATE_FUSE", 1);
    if (!enabled) {
        return false;
    }
    const ggml_tensor * wg = mm->src[0], * x = mm->src[1];
    if (sig->src[0] != mm || !(mul->src[0] == sig || mul->src[1] == sig) || !(add->src[0] == mul || add->src[1] == mul)) {
        return false;
    }
    const ggml_tensor * sh  = mul->src[0] == sig ? mul->src[1] : mul->src[0];
    const ggml_tensor * moe = add->src[0] == mul ? add->src[1] : add->src[0];
    const int64_t K = wg->ne[0], E = sh->ne[0], N = x->ne[1];
    // same-index elementwise tail (out may replace moe/shexp exactly); the dot inputs may not be overwritten
    // by another token's block
    const auto same_or_disjoint = [](const ggml_tensor * o, const ggml_tensor * in) {
        return !shexp_overlap(o, in) || (o->data == in->data && o->nb[1] == in->nb[1]);
    };
    // decode / verify only: at prefill the launches do not matter and the unfused path keeps prefill bit-identical
    return N <= 8 && (wg->type == GGML_TYPE_F32 || wg->type == GGML_TYPE_BF16) && x->type == GGML_TYPE_F32 && sh->type == GGML_TYPE_F32 && moe->type == GGML_TYPE_F32 &&
        add->type == GGML_TYPE_F32 && wg->ne[1] == 1 && ggml_is_contiguous(wg) && K % 4 == 0 &&
        x->ne[0] == K && x->nb[0] == sizeof(float) && x->nb[1] % 16 == 0 && ((uintptr_t) x->data) % 16 == 0 &&
        ((uintptr_t) wg->data) % 16 == 0 && x->ne[2] == 1 && x->ne[3] == 1 &&
        mm->ne[0] == 1 && mm->ne[1] == N && ggml_are_same_shape(sh, moe) && ggml_are_same_shape(add, sh) &&
        sh->ne[1] == N && sh->ne[2] == 1 && sh->ne[3] == 1 && sh->nb[0] == sizeof(float) && moe->nb[0] == sizeof(float) &&
        add->nb[0] == sizeof(float) && E > 0 &&
        !shexp_overlap(add, wg) && same_or_disjoint(add, x) && same_or_disjoint(add, sh) && same_or_disjoint(add, moe);
}

void ggml_cuda_shexp_gate_add(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * sig,
                              const ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * wg = mm->src[0], * x = mm->src[1];
    const ggml_tensor * sh  = mul->src[0] == sig ? mul->src[1] : mul->src[0];
    const ggml_tensor * moe = add->src[0] == mul ? add->src[1] : add->src[0];
    auto kern = wg->type == GGML_TYPE_BF16 ? shexp_gate_add<true> : shexp_gate_add<false>;
    kern<<<(unsigned) x->ne[1], 256, 0, ctx.stream()>>>(
        (const char *) wg->data, (const float *) x->data, (const float *) sh->data, (const float *) moe->data,
        (float *) add->data, (int) wg->ne[0], (int) sh->ne[0], x->nb[1]/sizeof(float), sh->nb[1]/sizeof(float),
        moe->nb[1]/sizeof(float), add->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}
