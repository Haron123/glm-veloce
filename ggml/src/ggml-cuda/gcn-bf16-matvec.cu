#include "gcn-bf16-matvec.cuh"

// Dense BF16 x F32 matvec for 1..8 tokens on GCN (gfx906). The generic MMVF path converts each bf16 weight to f32
// through a slow software path (gfx906 has no native bf16), reaching ~75 GB/s on the hyper-connection matrices
// (Qwen3.8-Flash-Next: 320 x 10240 and 10240 x 320 per layer, ~25% of decode time). Here the tokens' activations
// stay f32 in LDS (the same precision as MMVF: exact bf16 -> f32 weights, f32 products and sums; only the
// accumulation order differs), each wave owns R rows and streams them in lane-coalesced 16-byte loads, and a
// bf16 pair converts with a shift (bf16 is the top half of f32). KS waves can share a row group, each summing
// a slice of K, partial sums meeting in LDS. GGML_CUDA_GCN_BF16_MV=0 puts MMVF back.
template <int NT, int R, int KS, int NW>
__launch_bounds__(64*NW)
static __global__ void gcn_bf16_mv(const char * __restrict__ W, const float * __restrict__ y, float * __restrict__ dst,
        const int K, const int M, const int64_t w_stride, const int64_t y_stride, const int64_t d_stride) {
    extern __shared__ float bf16mv_ys[]; // NT rows of K floats
    __shared__ float red[KS > 1 ? NW*R*NT : 1]; // per-wave partial sums of the K slices
    const int nc = K / 8; // 16-byte chunks (8 bf16) per row
    for (int t = 0; t < NT; ++t) {
        for (int c = threadIdx.x; c < nc; c += 64*NW) {
            *(float4 *) (&bf16mv_ys[t*K + 8*c])     = *(const float4 *) (y + t*y_stride + 8*c);
            *(float4 *) (&bf16mv_ys[t*K + 8*c + 4]) = *(const float4 *) (y + t*y_stride + 8*c + 4);
        }
    }
    __syncthreads();

    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int ks   = wave % KS;                                // K slice of this wave
    const int row0 = (blockIdx.x*(NW/KS) + wave / KS)*R;       // first of the wave's R rows
    const bool live = row0 < M;                                // no early exit: KS > 1 meets at a barrier
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    const uint4 * wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int gr = min(row0 + r, M - 1); // rows past M read row M-1, never stored
        wr[r] = (const uint4 *) (W + (int64_t) gr*w_stride);
    }
    // the row's 1 KB steps (64 lanes x 16 B) split over KS waves; rows of 2^n bytes put every wave's same offset on
    // the same DRAM channels, so each wave starts its walk at a different step of its slice
    const int ns   = (nc + 63) / 64;
    const int s_lo = ks*ns/KS;
    const int sn   = (ks + 1)*ns/KS - s_lo;
    const int rot  = sn > 0 ? (row0 / R) % sn : 0;
    if (live) {
#pragma unroll 4
        for (int s0 = 0; s0 < sn; ++s0) {
            const int sc = s_lo + (s0 + rot < sn ? s0 + rot : s0 + rot - sn);
            const int c  = sc*64 + lane;
            if (c >= nc) {
                continue;
            }
            uint4 q[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                q[r] = wr[r][c];
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4 yv0 = *(const float4 *) (&bf16mv_ys[t*K + 8*c]);
                const float4 yv1 = *(const float4 *) (&bf16mv_ys[t*K + 8*c + 4]);
                const float yv[8] = {yv0.x, yv0.y, yv0.z, yv0.w, yv1.x, yv1.y, yv1.z, yv1.w};
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float s = acc[r][t];
                    const unsigned * qq = (const unsigned *) &q[r];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        // two bf16 per dword: the low half is the even element, the high half the odd one;
                        // bf16 -> f32 is the value's bits in the top half of the dword
                        const unsigned p = qq[u];
                        s = fmaf(__uint_as_float(p << 16),          yv[2*u],     s);
                        s = fmaf(__uint_as_float(p & 0xFFFF0000u),  yv[2*u + 1], s);
                    }
                    acc[r][t] = s;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) {
                v += __shfl_xor(v, off, 64);
            }
            if constexpr (KS == 1) {
                if (lane == r*NT + t && live && row0 + r < M) {
                    dst[t*d_stride + row0 + r] = v;
                }
            } else {
                if (lane == r*NT + t) {
                    red[wave*R*NT + r*NT + t] = v;
                }
            }
        }
    }
    if constexpr (KS > 1) {
        __syncthreads();
        const int j = lane; // (r, t) = (j / NT, j % NT)
        if (ks == 0 && j < R*NT && live && row0 + j / NT < M) {
            float v = 0.0f;
#pragma unroll
            for (int k = 0; k < KS; ++k) {
                v += red[(wave + k)*R*NT + j];
            }
            dst[j%NT*d_stride + row0 + j / NT] = v;
        }
    }
}

bool ggml_cuda_gcn_bf16_matvec_supported(int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    // opt-in: the kernel is faster than MMVF's bf16 path (+5.7% tg on Qwen3.8-Flash-Next 2x MI50) but the f32
    // accumulation order differs from MMVF, so greedy outputs can flip on near-tied tokens; the default keeps
    // MMVF's bit-exact behavior. GGML_CUDA_GCN_BF16_MV=1 enables it.
    static const int env = [] { const char * e = getenv("GGML_CUDA_GCN_BF16_MV"); return e ? atoi(e) : 0; }();
    const int cc = ggml_cuda_info().devices[device].cc;
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    // the activations stay f32 (as MMVF's bf16 path), so every precision mode is served
    return !(!env || !GGML_CUDA_CC_IS_GCN(cc) || src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || N < 1 || N > 8 || K % 8 != 0 || N*K*sizeof(float) > 48*1024 || M > INT_MAX ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src0->nb[0] != sizeof(ggml_bf16_t) || src0->nb[1] % 16 != 0 || ((uintptr_t) src0->data) % 16 != 0 ||
            src1->nb[0] != sizeof(float) || (N > 1 && src1->nb[1] % 16 != 0) || ((uintptr_t) src1->data) % 16 != 0 ||
            dst->nb[0] != sizeof(float) || (N > 1 && dst->nb[1] % sizeof(float) != 0));
}

bool ggml_cuda_gcn_bf16_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!ggml_cuda_gcn_bf16_matvec_supported(ctx.device, src0, src1, dst)) {
        return false;
    }
    static const int env_r  = [] { const char * e = getenv("GGML_CUDA_GCN_BF16_MV_R"); return e ? atoi(e) : 0; }();
    static const int env_ks = [] { const char * e = getenv("GGML_CUDA_GCN_BF16_MV_KS"); return e ? atoi(e) : 0; }();
    static const int env_nw = [] { const char * e = getenv("GGML_CUDA_GCN_BF16_MV_NW"); return e ? atoi(e) : 0; }();
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    const int r  = env_r ? env_r : N == 1 ? 1 : N == 2 ? 2 : M >= 4096 ? 2 : M >= 1024 ? 4 : 1;
    const int ks = env_ks ? env_ks : N == 1 && K >= 2048 ? 2 : 1;
    const int nw = env_nw ? env_nw : N <= 2 || M >= 4096 ? 4 : M >= 1024 || M < 512 ? 8 : 16;
    const int64_t ys = N > 1 ? src1->nb[1] / sizeof(float) : 0;
    const int64_t ds = N > 1 ? dst->nb[1] / sizeof(float) : 0;
    const size_t smem = (size_t) N*K*sizeof(float);
    cudaStream_t stream = ctx.stream();
#define BF16MV_GO(NT_, R_, KS_, NW_) do { const unsigned nb_ = (unsigned) ((M + ((NW_)/(KS_))*(R_) - 1)/(((NW_)/(KS_))*(R_))); \
        gcn_bf16_mv<NT_, R_, KS_, NW_><<<nb_, 64*(NW_), smem, stream>>>((const char *) src0->data, (const float *) src1->data, \
            (float *) dst->data, (int) K, (int) M, src0->nb[1], ys, ds); } while (0)
#define BF16MV(NT_, R_, KS_) do { if (nw == 16) { BF16MV_GO(NT_, R_, KS_, 16); } else if (nw == 8) { BF16MV_GO(NT_, R_, KS_, 8); } \
        else { BF16MV_GO(NT_, R_, KS_, 4); } } while (0)
#define BF16MV_KS(NT_, R_) switch (ks) { case 4: BF16MV(NT_, R_, 4); break; case 2: BF16MV(NT_, R_, 2); break; \
        default: BF16MV(NT_, R_, 1); break; }
#define BF16MV_R(NT_) switch (r) { case 4: BF16MV_KS(NT_, 4) break; case 2: BF16MV_KS(NT_, 2) break; default: BF16MV_KS(NT_, 1) break; }
    switch (N) {
        case 1:  BF16MV_R(1); break;
        case 2:  BF16MV_R(2); break;
        case 3:  BF16MV_R(3); break;
        case 4:  BF16MV_R(4); break;
        case 5:  BF16MV_R(5); break;
        case 6:  BF16MV_R(6); break;
        case 7:  BF16MV_R(7); break;
        default: BF16MV_R(8); break;
    }
#undef BF16MV_R
#undef BF16MV_KS
#undef BF16MV
#undef BF16MV_GO
    CUDA_CHECK(cudaGetLastError());
    return true;
}
