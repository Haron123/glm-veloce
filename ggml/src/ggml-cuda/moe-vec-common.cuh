#pragma once

#include "common.cuh"

// shared by moe-vec.cu and the per-type i-quant units (moe-iq.cuh, moe-iq-<type>.cu)

// decode without the expert sort: ids of the MUL_MAT_ID (row stride s1), activation row = slot / y_div
struct moe_direct_t {
    const int32_t * ids = nullptr;
    int64_t s1 = 0;
    int y_div = 1;
    // gate/up pairs: SwiGLU-clamp limit (GLM-5-Next: 10); INFINITY = plain SwiGLU (same arithmetic as before)
    float glu_limit = INFINITY;
};

// silu(min(g, limit)) * clamp(u, -limit, limit), as ggml_cuda_op_swiglu_clamp_single; with limit INFINITY exactly the plain
// SwiGLU expression these kernels used
static __device__ __forceinline__ float moe_glu(float g, float u, const float limit) {
    g = fminf(g, limit);
    u = fmaxf(fminf(u, limit), -limit);
    return g/(1.0f + expf(-g))*u;
}

// arguments of one i-quant MUL_MAT_ID decode launch (moe_coal_iq / moe_coal_iq_flat)
struct moe_iq_args {
    const char       * vg;      // gate (or the only) weights
    const char       * vu;      // up weights (= vg without a pair)
    const block_q8_1 * vy;      // q8_1 activations
    float            * dst;
    int64_t ne00, ne01;         // K, rows
    int64_t sg_row, sg_exp, su_row, su_exp; // byte strides of vg / vu
    int64_t n_used, ne12, n_slots;
    int64_t s1_dst, s2_dst, s_col;
    moe_direct_t dr;
    size_t  shm;                // dynamic LDS bytes
    int     max_grid;           // flat grid cap (blocks per CU * CUs)
    cudaStream_t stream;
};

// one explicit instance per type in moe-iq-<type>.cu; glu = gate/up pair (only the types with a GLU variant use it)
template <ggml_type type> void moe_iq_flat_launch(const moe_iq_args & a, bool glu, int lpr, int pf, bool sreg);
template <ggml_type type> void moe_iq_coal_launch(const moe_iq_args & a, bool glu, int lpr, int rg);

// moe_vec_q launcher: explicit instances in moe-vecq-<types>.cu
template <ggml_type type>
void moe_vec_launch_type(int NC, int T, dim3 grid, const char * x, const block_q8_1 * y, const int2 * tiles, const int32_t * n_tiles,
        const int32_t * bounds, const int32_t * ids_dst, float * dst, int ncols_x, int nrows_x, int64_t s01, int64_t s02,
        int64_t s_col_y, int n_used, int64_t s1, int64_t s2, cudaStream_t stream);
