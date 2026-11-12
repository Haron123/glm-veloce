#pragma once

#include "common.cuh"

// decode (1 token), GCN: the whole shared-expert tail of a Qwen3.8 Flash-Next FFN in two launches:
//   MUL_MAT(gate) MUL_MAT(up) GLU(SWIGLU) MUL_MAT(down) MUL_MAT(1-row shared gate) UNARY(SIGMOID) MUL ADD(moe)
// the gate and up matrices may be of different types. nodes = the 8 graph nodes in that order.
bool ggml_cuda_shexp_dec_supported(const ggml_tensor * const * nodes);
void ggml_cuda_shexp_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * const * nodes);
