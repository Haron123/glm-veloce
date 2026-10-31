#pragma once
#include "common.cuh"

// GCN dense BF16 x F32 matvec for 1..8 tokens (see gcn-bf16-matvec.cu); GGML_CUDA_GCN_BF16_MV=0 puts MMVF back
bool ggml_cuda_gcn_bf16_matvec_supported(int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_gcn_bf16_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
