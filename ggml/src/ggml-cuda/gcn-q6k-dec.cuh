#pragma once

#include "common.cuh"

// GCN single-token q6_K matvec: all weight loads of a lane in flight before the activations are staged and the math runs
bool ggml_cuda_gcn_q6k_dec_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_gcn_q6k_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
