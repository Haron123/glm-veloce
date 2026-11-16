#pragma once

#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

// LLAMA_QSA_KPOOL, default on (=0 disables): cache each block's pooled indexer key in a second key head, so the
// pool+norm+rope chain runs once per completed block instead of once per ubatch, and decode scores read the cached keys
// directly (fused GET_ROWS + MUL_MAT on CUDA/HIP). It measured neutral while per-token host input work hid it; with that
// fixed, tg at 64K context 45.6 -> 49.0 t/s (2x MI50). =0 restores the single-head buffer and the per-ubatch chain,
// bit-identical to before the cache existed.
bool llama_qsa_kpool_enabled();

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    //
    // with the pooled-key cache the caller passes the four pool tensors and leaves blk_cells and
    // blk_pos null, since no per-ubatch pool chain is built:
    //   pool_reps      I32 [n_blocks, ns]        stream-local cell whose head 1 holds each block's key
    //   new_pool_cells I32 [ratio*n_new_max, ns] members of the pools to (re)compute, slot order
    //   new_pool_reps  I64 [n_new_max*ns]        GLOBAL dest rows, the cpy_k_part convention
    //   upd_pos        I32 [4*n_new_max*ns]      mrope rows for the update rope, blk_pos layout
    // rebuild re-emits every complete block, not only those this ubatch closed. strm_of maps a
    // view stream to its physical stream, so a global row is strm_of[s]*kv_size + cell
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * pool_reps, ggml_tensor * new_pool_cells, ggml_tensor * new_pool_reps,
                       ggml_tensor * upd_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, bool rebuild, const uint32_t * strm_of) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // a pooled key is a pure function of its block's cells, so anything structural must re-pool
    // once. conservative on purpose: a needless rebuild costs one pass, a missed one serves a stale key
    void kpool_invalidate();

    // seq_rm of a tail [p0, end) with at most one sequence present keeps the pooled keys: it only removes members of
    // blocks that end at or after p0, those blocks leave the complete set, and the ubatch that completes one again
    // writes one of its positions, so it re-pools it (bit-identical). MTP rolls back every step: at 29K context
    // the full rebuild cost ~1.3 ms per step. LLAMA_QSA_KPOOL_TAIL=0: re-pool after every seq_rm
    bool kpool_tail_keeps(llama_seq_id seq_id, llama_pos p1) const;

    uint32_t n_seq_max_idx = 1;

    // the flat (one sequence) grouping of set_input_qsa, kept between calls: a decode step only fills cells, so the next
    // call updates the few blocks they touch instead of rescanning every cell (single stream only)
    struct qsa_inc_state {
        bool                   valid    = false;
        const void *           cells    = nullptr;
        int64_t                r        = 0;
        int64_t                n_kv     = 0;
        int64_t                n_blocks = 0;
        std::vector<llama_pos> pos;       // [n_kv] the cells' positions this state describes
        std::vector<uint64_t>  slots;     // [n_blocks] slots present per position block
        std::vector<int32_t>   first;     // [n_blocks] lowest cell of each position block
        std::vector<int32_t>   bid;       // [n_blocks] block id of a complete position block, else -1
        std::vector<int32_t>   members;   // [r*n_blocks] cell of each slot, -1 if absent
        std::vector<int32_t>   blk_of;    // [n_kv] block id of each cell, -1 if in no complete block
        std::vector<int32_t>   bid_idx;   // per block id: first position
        std::vector<int32_t>   bid_cell;  // per block id: lowest cell
        int64_t                pb_last  = -1; // last complete position block
    };
    mutable qsa_inc_state qsa_inc;

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * pool_reps, ggml_tensor * new_pool_cells, ggml_tensor * new_pool_reps,
                       ggml_tensor * upd_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, bool rebuild) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
