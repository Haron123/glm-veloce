#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cstring>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

// the pooled-key cache (default on, LLAMA_QSA_KPOOL=0 off): each block's pooled key is computed once and read from
// the cache at decode instead of re-pooled every token. Stored in f16, so rounding-class: 8K-window PPL on code 1.2077
// vs 1.2078; tg at 64K context 45.6 -> 49.0 t/s with the flat inputs and the fused score (2x MI50, 150 W)
bool llama_qsa_kpool_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_QSA_KPOOL");

        return !e || atoi(e) != 0;
    }();

    return enabled;
}

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with one key head of indexer_head_size, as llama_kv_cache_dsa shapes its own. the
        // pooled-key cache adds a SECOND head: head 0 holds the raw indexer keys, head 1 the pooled
        // key of the block whose last member is that cell
        const uint32_t n_head_kv = llama_qsa_kpool_enabled() ? 2 : 1;

        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), n_head_kv);
        // pooled-key cache: each head is padded by 16 values, so the pooled keys (one per `ratio` cells, 4: 2304 B apart
        // instead of 2048) do not all map to the same HBM channels. gfx906: 16384 pooled rows read in ~8 us instead of ~43
        // (LLAMA_QSA_KPOOL_PAD: the pad, default 16)
        static const int kpool_pad = [] { const char * e = getenv("LLAMA_QSA_KPOOL_PAD"); return e ? atoi(e) : 16; }();
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size + (llama_qsa_kpool_enabled() ? kpool_pad : 0);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        // the pooled keys are stored beside the raw ones, so the cache cannot be quantized
        ggml_type type_idx = type_k;
        if (llama_qsa_kpool_enabled() && ggml_is_quantized(type_idx)) {
            LLAMA_LOG_WARN("%s: indexer key cache stays %s rather than %s: it also holds the pooled keys\n",
                    __func__, ggml_type_name(GGML_TYPE_F16), ggml_type_name(type_idx));
            type_idx = GGML_TYPE_F16;
        }

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_idx, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::kpool_invalidate() {
    if (mem_idx && llama_qsa_kpool_enabled()) {
        mem_idx->set_kpool_dirty();
    }
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    kpool_invalidate();
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    kpool_invalidate();

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    kpool_invalidate();
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }

    kpool_invalidate();
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }

    kpool_invalidate();
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }

    kpool_invalidate();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

        // the restored cells need not be the ones the pooled keys were computed from
        kpool_invalidate();

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * pool_reps,
        ggml_tensor * new_pool_cells,
        ggml_tensor * new_pool_reps,
        ggml_tensor * upd_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        bool rebuild,
        const uint32_t * strm_of) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    // the pooled-key cache: a block's key is written once, when the block completes
    const bool kcache = pool_reps != nullptr;

    GGML_ASSERT((new_pool_cells != nullptr) == kcache);
    GGML_ASSERT((new_pool_reps  != nullptr) == kcache);
    GGML_ASSERT((upd_pos        != nullptr) == kcache);
    GGML_ASSERT(kcache == (blk_cells == nullptr) && "the pooled-key cache replaces the per-ubatch pool chain");
    GGML_ASSERT(kcache == (blk_pos   == nullptr));

    const int64_t n_blocks = kcache ? pool_reps->ne[0] : blk_pos->ne[0]/(4*n_ns);

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int64_t n_new_max = 0;

    if (kcache) {
        GGML_ASSERT(ggml_backend_buffer_is_host(pool_reps     ->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_cells->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_reps ->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(upd_pos       ->buffer));

        GGML_ASSERT(pool_reps     ->type == GGML_TYPE_I32);
        GGML_ASSERT(new_pool_cells->type == GGML_TYPE_I32);
        GGML_ASSERT(new_pool_reps ->type == GGML_TYPE_I64);
        GGML_ASSERT(upd_pos       ->type == GGML_TYPE_I32);

        GGML_ASSERT(pool_reps->ne[0] == n_blocks && pool_reps->ne[1] == n_ns);
        GGML_ASSERT(new_pool_cells->ne[0] % r == 0 && new_pool_cells->ne[1] == n_ns);

        n_new_max = new_pool_cells->ne[0]/r;

        GGML_ASSERT(new_pool_reps->ne[0] == n_new_max*n_ns);
        GGML_ASSERT(upd_pos->ne[0] == 4*n_new_max*n_ns);
        GGML_ASSERT(strm_of != nullptr);
    }

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = blk_cells ? (int32_t *) blk_cells->data : nullptr;
    int32_t * dst_blk_pos   = blk_pos   ? (int32_t *) blk_pos  ->data : nullptr;
    float   * dst_bias      = (float   *) bias->data;

    int32_t * dst_pool_reps = kcache ? (int32_t *) pool_reps     ->data : nullptr;
    int32_t * dst_new_cells = kcache ? (int32_t *) new_pool_cells->data : nullptr;
    int64_t * dst_new_reps  = kcache ? (int64_t *) new_pool_reps ->data : nullptr;
    int32_t * dst_upd_pos   = kcache ? (int32_t *) upd_pos       ->data : nullptr;

    // a global row is strm_of[s]*kv_size + cell, the convention set_input_k_idxs uses
    const int64_t kv_size = kcache ? (int64_t) get_mem_idx()->get_size() : 0;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // block and slot of a (non-negative) position: shift and mask for a power-of-two ratio. 64-bit division costs
    // ~60 cycles here and the scans below did two per cell (~2.5 ms per decode token at 64K context)
    const bool    r_pow2  = (r & (r - 1)) == 0;
    const int     r_shift = __builtin_ctzll((unsigned long long) r);
    const auto blk_div = [r_pow2, r_shift, r](int64_t idx) -> int64_t { return r_pow2 ? idx >> r_shift : idx/r; };
    const auto blk_mod = [r_pow2, r](int64_t idx) -> int64_t { return r_pow2 ? idx & (r - 1) : idx%r; };

    // runs per ubatch and is O(n_kv) per stream: ~0.7 ms per decode token at 64K context on the flat path below
    // (was ~3 ms: the group lists and two 64-bit divisions per cell). the scratch is reused across calls
    thread_local std::vector<int32_t> blk_of;
    thread_local std::vector<int32_t> cell_grp;
    thread_local std::vector<int32_t> grp_head;
    blk_of  .resize(n_kv);
    cell_grp.resize(n_kv);
    grp_head.resize(n_blocks);
    thread_local std::vector<uint64_t> flat_slots;
    thread_local std::vector<int32_t>  flat_first;
    thread_local std::vector<int32_t>  flat_bid;
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    // pooled-key cache only: the members of each block, its last member, whether this ubatch
    // writes one of its positions, and the mrope rows its pooled key is roped at
    std::vector<int32_t>   blk_cells_buf;
    std::vector<int32_t>   bid_last;
    std::vector<uint8_t>   bid_touched;
    std::vector<int32_t>   bid_sec_pos;
    std::vector<llama_pos> pos_set;

    if (kcache) {
        blk_cells_buf.resize(r*n_blocks);
        bid_last    .resize(n_blocks);
        bid_touched .resize(n_blocks);
        bid_sec_pos .resize(4*n_blocks);
    }

    if (dst_blk_pos) {
        std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        // no per-ubatch pool chain in kpool mode, so the members land in a scratch buffer
        int32_t * cur_blk_cells = dst_blk_cells ? dst_blk_cells + s*(r*n_blocks) : blk_cells_buf.data();

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        if (kcache) {
            std::fill(bid_last.begin(),    bid_last.end(),    -1);
            std::fill(bid_touched.begin(), bid_touched.end(),  0);

            // a block is touched when this ubatch writes one of its positions. the set is over the
            // stream's own tokens, so the cost is O(tokens), not O(n_kv)
            pos_set.clear();

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                pos_set.push_back(ubatch->pos[s*n_tps + ii]);
            }

            std::sort(pos_set.begin(), pos_set.end());
        }

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = blk_div(idx);

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << blk_mod(idx);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (blk_mod(idx) == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        // one sequence without repeated positions (every text chat): a cell's group is simply its block pos/r, so flat
        // per-block arrays replace the group lists (~2 ms per decode token at 64K context); same outputs
        // LLAMA_QSA_FLAT=0: always the general path
        static const bool flat_env = [] { const char * e = getenv("LLAMA_QSA_FLAT"); return !e || atoi(e) != 0; }();
        bool flat = flat_env && one_seq;

        // one stream: the grouping is kept between calls (qsa_inc) and a call that only filled cells updates the blocks
        // they touch; any other change (removed or moved cells, a repeated slot, a block completing out of order) or a
        // window that shrank takes the full scan, which refreshes the state (LLAMA_QSA_INC=0: always the full scan)
        static const bool inc_env = [] { const char * e = getenv("LLAMA_QSA_INC"); return !e || atoi(e) != 0; }();
        auto &        st      = qsa_inc;
        const bool    inc_use = flat && inc_env && n_ns == 1;
        bool          inc     = false;

        if (inc_use && st.valid && st.cells == (const void *) &cells && st.r == r && n_kv >= st.n_kv && n_blocks >= st.n_blocks) {
            inc = [&]() {
                const llama_pos * P = cells.pos_data();

                int32_t chg[64];
                int     n_chg = 0;

                constexpr int64_t CH = 1024;
                for (int64_t j0 = 0; j0 < st.n_kv; j0 += CH) {
                    const int64_t n = std::min(CH, st.n_kv - j0);
                    if (memcmp(P + j0, st.pos.data() + j0, n*sizeof(llama_pos)) == 0) {
                        continue;
                    }
                    for (int64_t j = j0; j < j0 + n; ++j) {
                        if (P[j] != st.pos[j]) {
                            if (n_chg == 64 || st.pos[j] != -1) {
                                return false;
                            }
                            chg[n_chg++] = (int32_t) j;
                        }
                    }
                }
                for (int64_t j = st.n_kv; j < n_kv; ++j) {
                    if (P[j] != -1) {
                        if (n_chg == 64) {
                            return false;
                        }
                        chg[n_chg++] = (int32_t) j;
                    }
                }
                for (int k = 0; k < n_chg; ++k) {
                    if (P[chg[k]] < 0 || blk_div(P[chg[k]]) >= n_blocks) {
                        return false;
                    }
                }

                st.pos    .resize(n_kv, -1);
                st.blk_of .resize(n_kv, -1);
                st.slots  .resize(n_blocks, 0);
                st.first  .resize(n_blocks, -1);
                st.bid    .resize(n_blocks, -1);
                st.members.resize(r*n_blocks, -1);
                st.n_kv     = n_kv;
                st.n_blocks = n_blocks;

                // in position order, so blocks complete in order
                std::sort(chg, chg + n_chg, [P](int32_t a, int32_t b) { return P[a] < P[b]; });

                for (int k = 0; k < n_chg; ++k) {
                    const int64_t  j   = chg[k];
                    const int64_t  p   = P[j];
                    const int64_t  pb  = blk_div(p);
                    const int64_t  sl  = blk_mod(p);
                    const uint64_t bit = uint64_t(1) << sl;

                    if (st.slots[pb] & bit) {
                        st.valid = false; // half applied: the full scan rebuilds it
                        return false;
                    }

                    st.slots[pb]        |= bit;
                    st.members[pb*r + sl] = (int32_t) j;
                    st.pos[j]             = (llama_pos) p;

                    if (st.first[pb] < 0 || j < st.first[pb]) {
                        st.first[pb] = (int32_t) j;
                    }

                    if (st.slots[pb] == slots_full) {
                        if (pb <= st.pb_last) {
                            st.valid = false; // an earlier block completed late: the block ids after it shift
                            return false;
                        }

                        const int32_t b = (int32_t) st.bid_idx.size();

                        st.bid[pb]  = b;
                        st.pb_last  = pb;

                        st.bid_idx .push_back((int32_t) (pb*r));
                        st.bid_cell.push_back(st.first[pb]);

                        for (int64_t m = 0; m < r; ++m) {
                            st.blk_of[st.members[pb*r + m]] = b;
                        }
                    }
                }

                return true;
            }();
        }

        if (flat && !inc) {
            flat_slots.assign(n_blocks, 0);
            flat_first.assign(n_blocks, -1);

            if (inc_use) {
                st.members.assign(r*n_blocks, -1);
            }

            // plain pointers: the thread_local vectors would be re-resolved (TLS lookup) inside the loop
            uint64_t * slots = flat_slots.data();
            int32_t  * first = flat_first.data();
            int32_t  * memb  = inc_use ? st.members.data() : nullptr;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = cells.pos_get(j);
                const int64_t pb  = blk_div(idx);

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                const uint64_t bit = uint64_t(1) << blk_mod(idx);

                dup |= (slots[pb] & bit) != 0;

                slots[pb] |= bit;

                if (first[pb] < 0) {
                    first[pb] = (int32_t) j;
                }

                if (memb) {
                    memb[pb*r + blk_mod(idx)] = (int32_t) j;
                }
            }

            // repeated positions of an image (mrope): the cells are ranked below, on the general path
            if (dup && ubatch->is_pos_2d()) {
                flat = false;
            }
        }

        if (!flat) {
            group_cells();
        }

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        if (inc) {
            n_bid = (int32_t) st.bid_idx.size();

            bid_idx  = st.bid_idx;
            bid_cell = st.bid_cell;
        } else if (flat) {
            flat_bid.resize(n_blocks);

            for (int64_t pb = 0; pb < n_blocks; ++pb) {
                flat_bid[pb] = -1;

                if (flat_slots[pb] != slots_full) {
                    continue;
                }

                flat_bid[pb] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(flat_first[pb]);
                bid_slot0.push_back(-1); // read only when ranked
            }
        } else {
            for (int64_t pb = 0; pb < n_blocks; ++pb) {
                for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                    if (grp_slots[g] != slots_full) {
                        continue;
                    }

                    grp_bid[g] = n_bid++;

                    bid_idx  .push_back((int32_t) (pb*r));
                    bid_cell .push_back(grp_first[g]);
                    bid_slot0.push_back(grp_slot0[g]);
                }
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            if (dst_blk_pos) {
                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
                }
            }

            if (kcache) {
                std::copy(sec_pos, sec_pos + 4, bid_sec_pos.begin() + 4*b);
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        int32_t       * blk_of_p = inc ? st.blk_of.data() : blk_of.data();
        const int32_t * fbid     = inc ? st.bid.data() : flat ? flat_bid.data() : nullptr;
        const int32_t * cgrp     = cell_grp.data();

        if (inc) {
            for (int64_t j = 0; j < n_kv; ++j) {
                cur_cell_blk[j] = blk_of_p[j] < 0 ? dead_bid : blk_of_p[j];
            }

            const int32_t * memb = st.members.data();

            for (int32_t b = 0; b < n_bid; ++b) {
                const int32_t * mb = memb + (int64_t) bid_idx[b]; // members of position block bid_idx[b]/r, r per block
                std::copy(mb, mb + r, cur_blk_cells + (int64_t) b*r);

                if (kcache) {
                    bid_last[b] = mb[r - 1];
                }
            }
        }

        for (int64_t j = 0; j < (inc ? 0 : n_kv); ++j) {
            if (flat) {
                const int64_t pb = cells.is_empty(j) ? n_blocks : blk_div(cells.pos_get(j));

                blk_of_p[j] = pb < n_blocks ? fbid[pb] : -1;
            } else {
                const int32_t g = cgrp[j];

                blk_of_p[j] = g < 0 ? -1 : grp_bid[g];
            }

            if (blk_of_p[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of_p[j]*r + blk_mod(idx)] = (int32_t) j;

                if (kcache) {
                    // a block's key lives in the row of its LAST member, and only a block this
                    // ubatch writes a position into can have changed
                    if (blk_mod(idx) == r - 1) {
                        bid_last[blk_of_p[j]] = (int32_t) j;
                    }

                    // flat: the touched blocks are those of the ubatch positions, marked below in O(tokens)
                    if (!flat && std::binary_search(pos_set.begin(), pos_set.end(), cells.pos_get(j))) {
                        bid_touched[blk_of_p[j]] = 1;
                    }
                }
            }

            cur_cell_blk[j] = blk_of_p[j] < 0 ? dead_bid : blk_of_p[j];
        }

        if (inc_use && !inc) {
            // refresh the kept state from this full scan; a repeated slot would not survive the round trip
            st.valid = flat && !dup && !oor;
            if (st.valid) {
                const llama_pos * P = cells.pos_data();

                st.cells    = (const void *) &cells;
                st.r        = r;
                st.n_kv     = n_kv;
                st.n_blocks = n_blocks;
                st.pos      .assign(P, P + n_kv);
                st.slots    .assign(flat_slots.begin(), flat_slots.begin() + n_blocks);
                st.first    .assign(flat_first.begin(), flat_first.begin() + n_blocks);
                st.bid      .assign(flat_bid.begin(),   flat_bid.begin()   + n_blocks);
                st.blk_of   .assign(blk_of_p, blk_of_p + n_kv);
                st.bid_idx  = bid_idx;
                st.bid_cell = bid_cell;
                st.pb_last  = n_bid > 0 ? bid_idx[n_bid - 1]/r : -1;
            }
        } else if (!inc_use && n_ns == 1) {
            st.valid = false;
        }

        // LLAMA_QSA_INC_CHECK=1 (debug): compare the incrementally kept grouping with a scan from scratch
        static const bool inc_check = [] { const char * e = getenv("LLAMA_QSA_INC_CHECK"); return e && atoi(e) != 0; }();
        if (inc && inc_check) {
            std::vector<uint64_t> sl(n_blocks, 0);
            std::vector<int32_t>  fi(n_blocks, -1), bi(n_blocks, -1);
            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    const int64_t pb = blk_div(cells.pos_get(j));
                    sl[pb] |= uint64_t(1) << blk_mod(cells.pos_get(j));
                    fi[pb]  = fi[pb] < 0 ? (int32_t) j : fi[pb];
                }
            }
            int32_t nb = 0;
            for (int64_t pb = 0; pb < n_blocks; ++pb) {
                if (sl[pb] == slots_full) {
                    GGML_ASSERT(nb < n_bid && bid_idx[nb] == pb*r && bid_cell[nb] == fi[pb] && "qsa inc: block list");
                    bi[pb] = nb++;
                }
            }
            GGML_ASSERT(nb == n_bid && "qsa inc: block count");
            for (int64_t j = 0; j < n_kv; ++j) {
                const int32_t e = cells.is_empty(j) ? -1 : bi[blk_div(cells.pos_get(j))];
                GGML_ASSERT(st.blk_of[j] == e && "qsa inc: cell block");
            }
        }

        if (kcache && flat) {
            // a block is touched when this ubatch writes one of its positions; on the flat path a position's block is
            // pos/r, so the ubatch's own positions name them (same set as the per-cell search of the general path)
            for (const llama_pos p : pos_set) {
                const int64_t pb = p >= 0 ? blk_div(p) : n_blocks;
                if (pb < n_blocks && fbid[pb] >= 0) {
                    bid_touched[fbid[pb]] = 1;
                }
            }
        }

        if (kcache) {
            int32_t * cur_pool_reps = dst_pool_reps + s*n_blocks;
            int32_t * cur_new_cells = dst_new_cells + s*(r*n_new_max);
            int64_t * cur_new_reps  = dst_new_reps  + s*n_new_max;
            // the update chain ropes [idx_dim, 1, n_new_max*n_ns], so a stream's rows start at
            // s*n_new_max and mrope reads pos[sec*(n_new_max*n_ns) + s*n_new_max + p]
            int32_t * cur_upd_pos   = dst_upd_pos   + s*n_new_max;

            const int64_t strm = strm_of[s];

            // the spare block holds the unpooled cells: a mean over [0,0,..] is raw row 0, roped at
            // position 0. recomputing it is idempotent, so padding with it is always safe
            auto emit_spare = [&](int64_t p) {
                std::fill(cur_new_cells + p*r, cur_new_cells + (p + 1)*r, 0);

                cur_new_reps[p] = strm*kv_size;

                for (int64_t sec = 0; sec < 4; ++sec) {
                    cur_upd_pos[sec*(n_new_max*n_ns) + p] = 0;
                }
            };

            // every complete block reads its own key, whether or not this ubatch rewrites it
            // rows past the complete blocks are never written below, and get_rows does not bound
            // check, so clear the whole row: the bias keeps those rows unselectable
            std::fill(cur_pool_reps, cur_pool_reps + n_blocks, 0);

            for (int32_t b = 0; b < n_bid; ++b) {
                cur_pool_reps[b] = bid_last[b];
            }

            int64_t n_new = 0;

            for (int32_t b = 0; b < n_bid; ++b) {
                if (!bid_touched[b] && !rebuild) {
                    continue;
                }

                // bounded while a sequence's ubatch tokens are a contiguous run (llama-batch.cpp
                // enforces); fail loudly, clamping would serve a stale key
                GGML_ASSERT(n_new < n_new_max && "qsa k-pool: more pools completed than the fixed bound");

                std::copy(cur_blk_cells + b*r, cur_blk_cells + (b + 1)*r, cur_new_cells + n_new*r);

                cur_new_reps[n_new] = strm*kv_size + bid_last[b];

                for (int64_t sec = 0; sec < 4; ++sec) {
                    cur_upd_pos[sec*(n_new_max*n_ns) + n_new] = bid_sec_pos[4*b + sec];
                }

                n_new++;
            }

            // the spare block's key is the spare pool's, which lives in cell 0
            if (have_dead) {
                GGML_ASSERT(n_new < n_new_max && "qsa k-pool: no room for the spare pool");

                cur_pool_reps[dead_bid] = 0;

                emit_spare(n_new++);
            }

            for (int64_t p = n_new; p < n_new_max; ++p) {
                emit_spare(p);
            }

            GGML_ASSERT(n_new <= n_new_max);
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * pool_reps,
        ggml_tensor * new_pool_cells,
        ggml_tensor * new_pool_reps,
        ggml_tensor * upd_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        bool rebuild) const {
    GGML_ASSERT(mem != nullptr);

    std::vector<uint32_t> strm_of;

    if (pool_reps) {
        const auto * idx = get_idx();

        GGML_ASSERT(idx != nullptr);

        strm_of.resize(idx->get_n_stream());

        for (uint32_t s = 0; s < strm_of.size(); ++s) {
            strm_of[s] = idx->get_strm(s);
        }
    }

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, pool_reps, new_pool_cells, new_pool_reps,
            upd_pos, bias, ubatch, ratio, blk_bias, rebuild, strm_of.empty() ? nullptr : strm_of.data());
}
