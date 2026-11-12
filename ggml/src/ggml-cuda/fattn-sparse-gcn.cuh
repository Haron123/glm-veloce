#pragma once

#include "common.cuh"

// Single-query (decode) flash attention for GCN that reads only the K/V rows the mask leaves visible, once per KV
// head for all of its query heads. See fattn-sparse-gcn.cu.
bool ggml_cuda_flash_attn_sparse_gcn_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_sparse_gcn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// decode (one query row): build fa's row list from the indexer top-k (width cells, -1 = none) and the attention mask
// (one row of n_kv), in place of the masked row the sparse attention would compact; false if fa will not take this path
bool ggml_cuda_spg_list_from_topk(ggml_backend_cuda_context & ctx, const ggml_tensor * fa, const int32_t * tk, int width,
                                  const ggml_tensor * mask);

// Prefill (many query rows) variant: the rows are cut into tiles of a few consecutive queries; each tile gathers the
// union of its rows' visible columns and the tile kernel runs on the tiles as sequences. See fattn-sparse-gcn.cu.
bool ggml_cuda_flash_attn_sparse_prefill_gcn_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_sparse_prefill_gcn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_DSV4_SPARSE_ATTN (GCN): gathered window + top-k cells, then the tile kernel
bool ggml_cuda_dsv4_sparse_attn_supported(const ggml_tensor * dst);
void ggml_cuda_dsv4_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// MQA on a 512-dim latent row used as both K and V (GLM-5-Next absorbed MLA), decode / verify: the DSV4 direct kernel
bool ggml_cuda_flash_attn_mqa512_gcn_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_mqa512_gcn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
