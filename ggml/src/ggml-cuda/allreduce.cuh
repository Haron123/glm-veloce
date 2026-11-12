#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the GPU device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);


// Graph-safe AllReduce for decode chains (2 GPUs with P2P, F32 tensors up to 1 MB): eligible() checks a tensor,
// launch() enqueues the in-place sum for GPU `rank` (0/1) on `stream`; capturable into a graph (device-side tokens).
bool ggml_cuda_ar_chain_eligible(const ggml_cuda_ar_pipeline * pipeline, const ggml_tensor * t);
void ggml_cuda_ar_chain_launch(ggml_cuda_ar_pipeline * pipeline, int rank, ggml_tensor * t, cudaStream_t stream);
