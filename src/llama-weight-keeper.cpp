#include "llama-weight-keeper.h"

#include "llama-impl.h"

#include "ggml-alloc.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

typedef bool (*ipc_export_t)(ggml_backend_buffer_t buffer, void * handle, uint64_t * checksum);
typedef ggml_backend_buffer_t (*ipc_import_t)(ggml_backend_buffer_type_t buft, const void * handle, size_t size, uint64_t checksum);

static constexpr char LWK_MAGIC[8] = { 'L', 'W', 'K', 'E', 'E', 'P', '0', '1' };

struct lwk_header {
    char     magic[8];
    int64_t  pid;
    uint32_t n_files;
    uint32_t n_entries;
};

static void * lwk_proc(ggml_backend_buffer_type_t buft, const char * name) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

static uint64_t lwk_fnv(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

// what the importer must reproduce: every tensor of the context at the same offset with the same shape
static uint64_t lwk_layout(ggml_context * ctx, ggml_backend_buffer_t buf) {
    const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(buf);
    uint64_t h = 0xcbf29ce484222325ull;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        const int64_t offs = t->buffer == buf ? (int64_t) ((const uint8_t *) t->data - base) : -1;
        const int32_t type = t->type;
        h = lwk_fnv(h, t->name, strlen(t->name));
        h = lwk_fnv(h, &type, sizeof(type));
        h = lwk_fnv(h, t->ne, sizeof(t->ne));
        h = lwk_fnv(h, &offs, sizeof(offs));
    }
    return h;
}

llama_weight_keeper * llama_weight_keeper::get() {
    static std::unique_ptr<llama_weight_keeper> keeper = [] {
        const char * p = getenv("LLAMA_WEIGHT_KEEPER");
        std::unique_ptr<llama_weight_keeper> k;
#ifndef _WIN32
        if (p && *p) {
            k.reset(new llama_weight_keeper());
            k->path = p;
            const char * e = getenv("LLAMA_WEIGHT_KEEPER_EXPORT");
            k->exporting = e && atoi(e) != 0;
        }
#else
        GGML_UNUSED(p);
#endif
        return k;
    }();
    return keeper.get();
}

void llama_weight_keeper::set_files(const llama_files & lfiles) {
    files.clear();
    // a process may load the model again after freeing it
    used.assign(entries.size(), false);
#ifndef _WIN32
    for (const auto & f : lfiles) {
        struct stat st;
        file_id id = {};
        if (fstat(f->file_id(), &st) == 0) {
            id.dev      = st.st_dev;
            id.ino      = st.st_ino;
            id.size     = st.st_size;
            id.mtime_ns = (int64_t) st.st_mtim.tv_sec*1000000000 + st.st_mtim.tv_nsec;
        }
        files.push_back(id);
    }
#else
    GGML_UNUSED(lfiles);
#endif
}

bool llama_weight_keeper::load_manifest() {
    if (loaded) {
        return usable;
    }
    loaded = true;

    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(path.c_str(), "rb"), &fclose);
    if (!f) {
        LLAMA_LOG_INFO("%s: no keeper manifest at %s, loading from disk\n", __func__, path.c_str());
        return false;
    }
    lwk_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, f.get()) != 1 || memcmp(hdr.magic, LWK_MAGIC, sizeof(LWK_MAGIC)) != 0) {
        LLAMA_LOG_WARN("%s: %s is not a keeper manifest\n", __func__, path.c_str());
        return false;
    }
#ifndef _WIN32
    if (kill((pid_t) hdr.pid, 0) != 0 && errno != EPERM) {
        LLAMA_LOG_INFO("%s: keeper pid %lld is gone, loading from disk\n", __func__, (long long) hdr.pid);
        return false;
    }
#endif
    std::vector<file_id> mfiles(hdr.n_files);
    entries.resize(hdr.n_entries);
    if (fread(mfiles.data(), sizeof(file_id), mfiles.size(), f.get()) != mfiles.size() ||
        fread(entries.data(), sizeof(entry), entries.size(), f.get()) != entries.size()) {
        LLAMA_LOG_WARN("%s: truncated keeper manifest %s\n", __func__, path.c_str());
        entries.clear();
        return false;
    }
    if (mfiles.size() != files.size() || memcmp(mfiles.data(), files.data(), files.size()*sizeof(file_id)) != 0) {
        LLAMA_LOG_INFO("%s: the keeper holds other model files, loading from disk\n", __func__);
        entries.clear();
        return false;
    }
    used.assign(entries.size(), false);
    usable = true;
    LLAMA_LOG_INFO("%s: keeper pid %lld offers %zu buffers\n", __func__, (long long) hdr.pid, entries.size());
    return true;
}

// import the entry named `name` for the tensors of ctx into a buffer of buft (placed as ggml_backend_alloc_ctx_tensors_from_buft
// does), or nullptr (ctx untouched)
ggml_backend_buffer_t llama_weight_keeper::import_entry(ggml_context * ctx, ggml_backend_buffer_type_t buft, const char * name) {
    auto import = (ipc_import_t) lwk_proc(buft, "ggml_backend_buffer_ipc_import");
    if (!import) {
        return nullptr;
    }
    // a context that would span several buffers is not shared
    const size_t size = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
    if (size == 0 || size > ggml_backend_buft_get_max_size(buft)) {
        return nullptr;
    }
    for (size_t i = 0; i < entries.size(); ++i) {
        const entry & e = entries[i];
        if (used[i] || e.size != size || strncmp(e.buft, name, sizeof(e.buft)) != 0) {
            continue;
        }
        ggml_backend_buffer_t buf = import(buft, e.handle, e.size, e.checksum);
        if (!buf) {
            return nullptr;
        }

        // place the tensors as ggml_backend_alloc_ctx_tensors_from_buft does
        bool ok = true;
        ggml_tallocr talloc = ggml_tallocr_new(buf);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr && ok; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == nullptr && t->view_src == nullptr) {
                ok = ggml_tallocr_alloc(&talloc, t) == GGML_STATUS_SUCCESS;
            } else if (t->view_src != nullptr && t->buffer == nullptr) {
                ok = ggml_backend_view_init(t) == GGML_STATUS_SUCCESS;
            }
        }
        if (!ok || lwk_layout(ctx, buf) != e.layout) {
            LLAMA_LOG_WARN("%s: %s: tensor layout differs from the keeper's, loading from disk\n", __func__, name);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                if (t->buffer == buf) {
                    t->data   = nullptr;
                    t->buffer = nullptr;
                }
            }
            ggml_backend_buffer_free(buf);
            return nullptr;
        }
        used[i] = true;
        LLAMA_LOG_INFO("%s: %s: mapped %.2f MiB of weights from the keeper\n", __func__, name, size/1024.0/1024.0);
        return buf;
    }
    return nullptr;
}

// the manifest name of device i's part of a tensor-parallel (meta) buffer
static std::string lwk_meta_name(ggml_backend_buffer_type_t simple_buft, size_t i) {
    return std::string(ggml_backend_buft_name(simple_buft)) + "|m" + std::to_string(i);
}

ggml_backend_buffer_t llama_weight_keeper::import_ctx(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    if (exporting || !load_manifest()) {
        return nullptr;
    }
    if (ggml_backend_buft_is_meta(buft)) {
        // -sm tensor: every device's part is a separate entry; all of them or none
        auto get_simple = [](size_t i, ggml_context * sctx, ggml_backend_buffer_type_t sbuft, void * ud) -> ggml_backend_buffer_t {
            auto * self = (llama_weight_keeper *) ud;
            return self->import_entry(sctx, sbuft, lwk_meta_name(sbuft, i).c_str());
        };
        const std::vector<bool> used_before = used;
        ggml_backend_buffer_t buf = ggml_backend_meta_alloc_ctx_tensors_with(ctx, buft, get_simple, this);
        if (!buf) {
            used = used_before; // device parts imported before a failing one were released with the meta buffer
        }
        return buf;
    }
    return import_entry(ctx, buft, ggml_backend_buft_name(buft));
}

void llama_weight_keeper::add_export(ggml_context * ctx, ggml_backend_buffer_t buf) {
    if (ggml_backend_buffer_is_meta(buf)) {
        for (size_t i = 0; i < ggml_backend_meta_buffer_n_simple(buf); ++i) {
            ggml_backend_buffer_t sb  = ggml_backend_meta_buffer_get_simple(buf, i);
            ggml_context *        sct = ggml_backend_meta_buffer_get_simple_ctx(buf, i);
            if (sb && sct && ggml_backend_buffer_get_size(sb) > 0) {
                add_export_named(sct, sb, lwk_meta_name(ggml_backend_buffer_get_type(sb), i).c_str());
            }
        }
        return;
    }
    add_export_named(ctx, buf, ggml_backend_buft_name(ggml_backend_buffer_get_type(buf)));
}

void llama_weight_keeper::add_export_named(ggml_context * ctx, ggml_backend_buffer_t buf, const char * name) {
    auto exp = (ipc_export_t) lwk_proc(ggml_backend_buffer_get_type(buf), "ggml_backend_buffer_ipc_export");
    if (!exp) {
        return;
    }
    entry e = {};
    snprintf(e.buft, sizeof(e.buft), "%s", name);
    e.size   = ggml_backend_buffer_get_size(buf);
    e.layout = lwk_layout(ctx, buf);
    if (!exp(buf, e.handle, &e.checksum)) {
        LLAMA_LOG_WARN("%s: %s: cannot export\n", __func__, e.buft);
        return;
    }
    entries.push_back(e);
}

void llama_weight_keeper::serve() {
#ifndef _WIN32
    const std::string tmp = path + ".tmp";
    {
        std::unique_ptr<FILE, decltype(&fclose)> f(fopen(tmp.c_str(), "wb"), &fclose);
        if (!f) {
            LLAMA_LOG_ERROR("%s: cannot write %s\n", __func__, tmp.c_str());
            exit(1);
        }
        lwk_header hdr = {};
        memcpy(hdr.magic, LWK_MAGIC, sizeof(LWK_MAGIC));
        hdr.pid       = getpid();
        hdr.n_files   = files.size();
        hdr.n_entries = entries.size();
        fwrite(&hdr, sizeof(hdr), 1, f.get());
        fwrite(files.data(), sizeof(file_id), files.size(), f.get());
        fwrite(entries.data(), sizeof(entry), entries.size(), f.get());
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        LLAMA_LOG_ERROR("%s: cannot write %s\n", __func__, path.c_str());
        exit(1);
    }
    uint64_t total = 0;
    for (const auto & e : entries) {
        total += e.size;
    }
    LLAMA_LOG_INFO("%s: weight keeper ready: %zu buffers, %.2f GiB, pid %d, manifest %s\n",
            __func__, entries.size(), total/1024.0/1024.0/1024.0, (int) getpid(), path.c_str());
    fflush(stderr);

    // the host program's handlers would wait for a shutdown that never comes
    signal(SIGINT,  SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    while (true) {
        pause();
    }
#else
    exit(1);
#endif
}
