#include "shexp-dec.cuh"
#include "mmvq.cuh"
#include "vecdotq.cuh"

#include <cstdlib>

// The shared expert of a decode step reads ~2.6 MB of weights but ran as seven launches (gate, up, GLU, activation
// quantization, down, the 1-row gate, the gate tail), each latency-bound: ~41 us per layer on gfx906. Here:
//   shexp_gu_dec:   gate and up rows (one wave per row, each wave its matrix's type) and the shared gate's dot
//   shexp_down_dec: every block forms h = silu(g)*u and its q8_1 copy in LDS (640 values), then the down rows
//                   (one wave per row) and out = moe + sigmoid(gate dot) * down
// The row dots are MMVQ's vec_dot functions over the shared q8_1 copy of the input.

template <ggml_type type>
static constexpr __host__ __device__ int shexp_vdr() {
    return type == GGML_TYPE_Q4_K   ? VDR_Q4_K_Q8_1_MMVQ   : type == GGML_TYPE_Q5_K  ? VDR_Q5_K_Q8_1_MMVQ  :
           type == GGML_TYPE_Q6_K   ? VDR_Q6_K_Q8_1_MMVQ   : type == GGML_TYPE_IQ4_XS ? VDR_IQ4_XS_Q8_1_MMVQ :
           type == GGML_TYPE_IQ3_S  ? VDR_IQ3_S_Q8_1_MMVQ  : type == GGML_TYPE_IQ4_NL ? VDR_IQ4_NL_Q8_1_MMVQ :
           VDR_Q2_0_Q8_1_MMVQ;
}

template <ggml_type type>
static __device__ __forceinline__ float shexp_vec_dot(const void * vbq, const block_q8_1 * bq8_1, const int & kbx, const int & iqs) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        return vec_dot_q4_K_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        return vec_dot_q5_K_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        return vec_dot_q6_K_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        return vec_dot_iq4_xs_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        return vec_dot_iq3_s_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        return vec_dot_iq4_nl_q8_1(vbq, bq8_1, kbx, iqs);
    } else {
        static_assert(type == GGML_TYPE_Q2_0, "unsupported type");
        return vec_dot_q2_0_q8_1(vbq, bq8_1, kbx, iqs);
    }
}

// one wave: the dot of a row of nb blocks with the q8_1 activation, MMVQ's lane-to-chunk order; the sum in every lane
template <ggml_type type>
static __device__ __forceinline__ float shexp_row_dot(const char * __restrict__ row, const block_q8_1 * __restrict__ y, const int nb,
                                                      const int lane) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = shexp_vdr<type>();
    constexpr int bpi = vdr*WARP_SIZE/qi; // blocks per iteration of the wave
    float sum = 0.0f;
    for (int kbx = lane/(qi/vdr); kbx < nb; kbx += bpi) {
        sum += shexp_vec_dot<type>(row, &y[kbx*(qk/QK8_1)], kbx, vdr*(lane % (qi/vdr)));
    }
    return warp_reduce_sum<WARP_SIZE>(sum);
}

// the same for nt (2..4) tokens whose q8_1 rows are sy blocks apart: a row's weight loads and decode are shared by the
// tokens (the guarded per-token dots reuse token 0's loads); the sums go to v[0 .. nt-1], in every lane
template <ggml_type type>
static __device__ __forceinline__ void shexp_row_dot_n(const char * __restrict__ row, const block_q8_1 * __restrict__ y,
                                                       const int64_t sy, const int nb, const int lane, const int nt, float v[4]) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = shexp_vdr<type>();
    constexpr int bpi = vdr*WARP_SIZE/qi;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int kbx = lane/(qi/vdr); kbx < nb; kbx += bpi) {
        const int iqs = vdr*(lane % (qi/vdr));
        const block_q8_1 * yk = y + kbx*(qk/QK8_1);
        s0 += shexp_vec_dot<type>(row, yk, kbx, iqs);
        if (nt > 1) {
            s1 += shexp_vec_dot<type>(row, yk + sy, kbx, iqs);
        }
        if (nt > 2) {
            s2 += shexp_vec_dot<type>(row, yk + 2*sy, kbx, iqs);
        }
        if (nt > 3) {
            s3 += shexp_vec_dot<type>(row, yk + 3*sy, kbx, iqs);
        }
    }
    v[0] = warp_reduce_sum<WARP_SIZE>(s0);
    v[1] = nt > 1 ? warp_reduce_sum<WARP_SIZE>(s1) : 0.0f;
    v[2] = nt > 2 ? warp_reduce_sum<WARP_SIZE>(s2) : 0.0f;
    v[3] = nt > 3 ? warp_reduce_sum<WARP_SIZE>(s3) : 0.0f;
}

// blocks 0 .. 2R/4 - 1: 4 rows each (gate rows first, then up rows); the last block: the shared gate's dot
// NT tokens (1, or 2..4 at an MTP verify): token t's gate / up values at gu + t*2R, its shared gate dot at gdot[t]; the
// rows are decoded once for all tokens (MULTI, nt > 1)
template <ggml_type tg, ggml_type tu, bool MULTI>
__launch_bounds__(256)
static __global__ void shexp_gu_dec(
        const char * __restrict__ Wg, const char * __restrict__ Wu, const int64_t sg, const int64_t su, const int R, const int K,
        const block_q8_1 * __restrict__ y, const int64_t sy, const char * __restrict__ w1, const bool w1_bf16,
        const float * __restrict__ x, const int64_t sx, float * __restrict__ gu, float * __restrict__ gdot, const int NT) {
    const int wave = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (blockIdx.x == gridDim.x - 1) {
        __shared__ float red[256/WARP_SIZE];
#pragma unroll 1
        for (int t = 0; t < NT; ++t) {
            float acc = 0.0f;
            for (int k = threadIdx.x; k < K; k += blockDim.x) {
                const float w = w1_bf16 ? __uint_as_float((uint32_t) ((const uint16_t *) w1)[k] << 16) : ((const float *) w1)[k];
                acc += w*x[t*sx + k];
            }
            acc = warp_reduce_sum<WARP_SIZE>(acc);
            if (lane == 0) {
                red[wave] = acc;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < 256/WARP_SIZE; ++i) {
                    s += red[i];
                }
                gdot[t] = s;
            }
            __syncthreads();
        }
        return;
    }
    const int r = blockIdx.x*(256/WARP_SIZE) + wave;
    if (r >= 2*R) {
        return;
    }
    if constexpr (MULTI) {
        float v[4];
        if (r < R) {
            shexp_row_dot_n<tg>(Wg + r*sg, y, sy, K/ggml_cuda_type_traits<tg>::qk, lane, NT, v);
        } else {
            shexp_row_dot_n<tu>(Wu + (r - R)*su, y, sy, K/ggml_cuda_type_traits<tu>::qk, lane, NT, v);
        }
        if (lane == 0) {
            gu[r] = v[0];
#pragma unroll
            for (int t = 1; t < 4; ++t) {
                if (t < NT) {
                    gu[t*2*R + r] = v[t];
                }
            }
        }
    } else {
        float v;
        if (r < R) {
            v = shexp_row_dot<tg>(Wg + r*sg, y, K/ggml_cuda_type_traits<tg>::qk, lane);
        } else {
            v = shexp_row_dot<tu>(Wu + (r - R)*su, y, K/ggml_cuda_type_traits<tu>::qk, lane);
        }
        if (lane == 0) {
            gu[r] = v;
        }
    }
}

// one row per wave in 1024-thread blocks: h and its q8_1 copy are formed once per 16 rows, not once per 4 (with 256-thread
// blocks that prologue, repeated by ~11 blocks per CU, cost ~4 us of the 15.4 us per call; now 11.7 us). Several rows per
// wave were slower: each extra row adds a dependent round trip (2 rows 13.6 us, 3 rows 17.9 us)
#define SHEXP_DOWN_NT 1024
template <ggml_type td, bool MULTI>
__launch_bounds__(SHEXP_DOWN_NT)
static __global__ void shexp_down_dec(
        const char * __restrict__ Wd, const int64_t sd, const int M, const int R, const float * __restrict__ gu,
        const float * __restrict__ gdot, const float * __restrict__ moe, float * __restrict__ out,
        const float * __restrict__ experts, const float * __restrict__ ew, const int n_used, const int NT) {
    extern __shared__ block_q8_1 shexp_hq[]; // NT x R/32 blocks
    const int wave = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    // h = silu(g)*u and its q8_1 copy per token, quantized as quantize_q8_1 does (groups of 32 lanes = one block)
    for (int t = 0; t < NT; ++t) {
        const float * gut = gu + t*2*R;
        block_q8_1 * hq = shexp_hq + t*(R/QK8_1);
        for (int i0 = 0; i0 < R; i0 += blockDim.x) {
            const int i = i0 + threadIdx.x;
            const float g = i < R ? gut[i] : 0.0f;
            const float h = i < R ? g/(1.0f + expf(-g))*gut[R + i] : 0.0f;
            float amax = fabsf(h);
            float sum  = h;
            amax = warp_reduce_max<QK8_1>(amax);
            sum  = warp_reduce_sum<QK8_1>(sum);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(h / d);
            if (i < R) {
                hq[i / QK8_1].qs[i % QK8_1] = q;
                if (i % QK8_1 == 0) {
                    hq[i / QK8_1].ds = make_half2(d, sum);
                }
            }
        }
    }
    __syncthreads();
    const int r = blockIdx.x*(SHEXP_DOWN_NT/WARP_SIZE) + wave;
    if (r >= M) {
        return;
    }
    if constexpr (MULTI) {
        float v[4];
        shexp_row_dot_n<td>(Wd + r*sd, shexp_hq, R/QK8_1, R/ggml_cuda_type_traits<td>::qk, lane, NT, v);
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (t < NT) {
                    out[t*M + r] = moe[t*M + r] + v[t]*(1.0f/(1.0f + expf(-gdot[t])));
                }
            }
        }
        return;
    }
    {
        const int t = 0;
        const float v = shexp_row_dot<td>(Wd + r*sd, shexp_hq, R/ggml_cuda_type_traits<td>::qk, lane);
        if (lane == 0) {
            const float s = 1.0f/(1.0f + expf(-gdot[t]));
            float m;
            if (experts) {
                // the routed experts' weighted sum, as moe_weighted_reduction forms it (0 + sum of x*w in expert order)
                m = 0.0f;
                for (int e = 0; e < n_used; ++e) {
                    m += experts[(int64_t) e*M + r]*ew[e];
                }
            } else {
                m = moe[t*M + r];
            }
            out[t*M + r] = m + v*s;
        }
    }
}

static bool shexp_gu_type_ok(const ggml_type t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ4_XS || t == GGML_TYPE_IQ3_S;
}

bool ggml_cuda_shexp_dec_supported(const ggml_tensor * const * nodes) {
    // opt-in: measured neutral in the model (57.18 vs 57.20 t/s), a rounding change
    // default on since 2026-11-07 (+1.5% tg once the other launches were gone; rounding class: wording-level greedy diffs);
    // GGML_CUDA_SHEXP_DEC=0 off
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_SHEXP_DEC"); return !e || atoi(e) != 0; }();
    const ggml_tensor * gate = nodes[0], * up = nodes[1], * glu = nodes[2], * down = nodes[3];
    const ggml_tensor * sg = nodes[4], * sig = nodes[5], * mul = nodes[6], * add = nodes[7];
    if (!enabled || !GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    if (gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU || down->op != GGML_OP_MUL_MAT ||
            sg->op != GGML_OP_MUL_MAT || sig->op != GGML_OP_UNARY || mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD) {
        return false;
    }
    const ggml_tensor * x = gate->src[1];
    const ggml_tensor * wg = gate->src[0], * wu = up->src[0], * wd = down->src[0], * w1 = sg->src[0];
    const int64_t K = wg->ne[0], R = wg->ne[1];
    const bool w1_ok = (w1->type == GGML_TYPE_F32 || w1->type == GGML_TYPE_BF16) && ggml_is_contiguous(w1) && w1->ne[0] == K &&
        ggml_nrows(w1) == 1;
    const ggml_tensor * moe = add->src[0] == mul ? add->src[1] : add->src[0];
    // 2..4 tokens (an MTP verify): pp3 batch 27.31 -> 26.99 ms; GGML_CUDA_SHEXP_NT=0 keeps those on the generic path
    static const bool multi = [] { const char * e = getenv("GGML_CUDA_SHEXP_NT"); return !e || atoi(e) != 0; }();
    const int64_t nt = x->ne[1];
    return up->src[1] == x && (nt == 1 || multi) && sg->src[1] == x && x->type == GGML_TYPE_F32 && nt >= 1 && nt <= 4 && x->ne[2] == 1 && x->ne[3] == 1 &&
        x->ne[0] == K && ggml_is_contiguous(x) &&
        ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU && glu->src[0] == gate && glu->src[1] == up &&
        down->src[1] == glu && ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID && sig->src[0] == sg &&
        ((mul->src[0] == down && mul->src[1] == sig) || (mul->src[0] == sig && mul->src[1] == down)) &&
        (add->src[0] == mul || add->src[1] == mul) && w1_ok &&
        shexp_gu_type_ok(wg->type) && shexp_gu_type_ok(wu->type) && (wd->type == GGML_TYPE_IQ4_NL || wd->type == GGML_TYPE_Q2_0) &&
        ggml_is_contiguous(wg) && ggml_is_contiguous(wu) && ggml_is_contiguous(wd) && wg->ne[2] == 1 && wu->ne[2] == 1 &&
        wd->ne[2] == 1 && wu->ne[0] == K && wu->ne[1] == R && wd->ne[0] == R && K % QK_K == 0 && R % 64 == 0 &&
        R <= 4096 && moe->type == GGML_TYPE_F32 && ggml_is_contiguous(moe) && ggml_nelements(moe) == wd->ne[1]*nt &&
        add->type == GGML_TYPE_F32 && ggml_is_contiguous(add) && ggml_nelements(add) == wd->ne[1]*nt;
}

void ggml_cuda_shexp_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * const * nodes) {
    const ggml_tensor * gate = nodes[0], * up = nodes[1], * down = nodes[3], * sg = nodes[4], * mul = nodes[6];
    ggml_tensor * add = (ggml_tensor *) nodes[7];
    const ggml_tensor * x  = gate->src[1];
    const ggml_tensor * wg = gate->src[0], * wu = up->src[0], * wd = down->src[0], * w1 = sg->src[0];
    const ggml_tensor * moe = add->src[0] == mul ? add->src[1] : add->src[0];
    const int K = (int) wg->ne[0], R = (int) wg->ne[1], M = (int) wd->ne[1];
    cudaStream_t stream = ctx.stream();
    // the routed experts' weighted reduction was left to this kernel (see the MUL handler in ggml-cuda.cu)
    const float * experts = nullptr, * ew = nullptr;
    int n_used = 0;
    if (ctx.shexp_moe.dst == moe) {
        experts = (const float *) ctx.shexp_moe.experts->data;
        ew      = (const float *) ctx.shexp_moe.weights->data;
        n_used  = ctx.shexp_moe.n_used;
        ctx.shexp_moe = {};
    }

    ggml_cuda_pool_alloc<char> yq_local(ctx.pool());
    const block_q8_1 * yq = (const block_q8_1 *) ggml_cuda_q8_1_activation(ctx, x, wg->type, GGML_PAD(K, MATRIX_ROW_PADDING), yq_local);
    const int nt = (int) x->ne[1];
    const int64_t sy = GGML_PAD(K, MATRIX_ROW_PADDING)/QK8_1, sx = x->nb[1]/sizeof(float);
    ggml_cuda_pool_alloc<float> gu(ctx.pool(), (size_t) (2*R + 1)*nt);
    float * gdot = gu.get() + 2*R*nt;

    const int nb1 = (2*R + 3)/4 + 1;
#define SHEXP_GU(TG, TU) if (nt > 1) SHEXP_GU1(TG, TU, true); else SHEXP_GU1(TG, TU, false)
#define SHEXP_GU1(TG, TU, MU) shexp_gu_dec<TG, TU, MU><<<nb1, 256, 0, stream>>>((const char *) wg->data, (const char *) wu->data, wg->nb[1], \
        wu->nb[1], R, K, yq, sy, (const char *) w1->data, w1->type == GGML_TYPE_BF16, (const float *) x->data, sx, gu.get(), gdot, nt)
#define SHEXP_GU_U(TG) switch (wu->type) { \
        case GGML_TYPE_Q4_K:   SHEXP_GU(TG, GGML_TYPE_Q4_K);   break; \
        case GGML_TYPE_Q5_K:   SHEXP_GU(TG, GGML_TYPE_Q5_K);   break; \
        case GGML_TYPE_Q6_K:   SHEXP_GU(TG, GGML_TYPE_Q6_K);   break; \
        case GGML_TYPE_IQ4_XS: SHEXP_GU(TG, GGML_TYPE_IQ4_XS); break; \
        default:               SHEXP_GU(TG, GGML_TYPE_IQ3_S);  break; }
    switch (wg->type) {
        case GGML_TYPE_Q4_K:   SHEXP_GU_U(GGML_TYPE_Q4_K)   break;
        case GGML_TYPE_Q5_K:   SHEXP_GU_U(GGML_TYPE_Q5_K)   break;
        case GGML_TYPE_Q6_K:   SHEXP_GU_U(GGML_TYPE_Q6_K)   break;
        case GGML_TYPE_IQ4_XS: SHEXP_GU_U(GGML_TYPE_IQ4_XS) break;
        default:               SHEXP_GU_U(GGML_TYPE_IQ3_S)  break;
    }
#undef SHEXP_GU_U
#undef SHEXP_GU
#undef SHEXP_GU1
    CUDA_CHECK(cudaGetLastError());

    const size_t smem = (size_t) nt*(R/QK8_1)*sizeof(block_q8_1);
    const int nb2 = (M + SHEXP_DOWN_NT/WARP_SIZE - 1)/(SHEXP_DOWN_NT/WARP_SIZE);
#define SHEXP_DN(TD, MU) shexp_down_dec<TD, MU><<<nb2, SHEXP_DOWN_NT, smem, stream>>>((const char *) wd->data, wd->nb[1], M, R, \
        gu.get(), gdot, (const float *) moe->data, (float *) add->data, experts, ew, n_used, nt)
    if (wd->type == GGML_TYPE_IQ4_NL) {
        if (nt > 1) SHEXP_DN(GGML_TYPE_IQ4_NL, true); else SHEXP_DN(GGML_TYPE_IQ4_NL, false);
    } else {
        if (nt > 1) SHEXP_DN(GGML_TYPE_Q2_0, true); else SHEXP_DN(GGML_TYPE_Q2_0, false);
    }
#undef SHEXP_DN
    CUDA_CHECK(cudaGetLastError());
}
