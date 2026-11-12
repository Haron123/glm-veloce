#pragma once

#include "llama-mmap.h"

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

// Keeps the device weights in VRAM across process restarts, so a restart skips the disk read:
//   LLAMA_WEIGHT_KEEPER=<manifest path>     map the weights of a running keeper when they match this load
//   LLAMA_WEIGHT_KEEPER_EXPORT=1 with it    this process is the keeper: export the weights after the load, then block
// A buffer is imported only if the model files, its buffer type, size, tensor layout and contents match the export;
// anything else is allocated and loaded as usual.
struct llama_weight_keeper {
    static llama_weight_keeper * get(); // nullptr when LLAMA_WEIGHT_KEEPER is unset

    bool exporting = false;

    void set_files(const llama_files & files);

    // the tensors of ctx placed in an imported buffer, or nullptr
    ggml_backend_buffer_t import_ctx(ggml_context * ctx, ggml_backend_buffer_type_t buft);

    void add_export(ggml_context * ctx, ggml_backend_buffer_t buf);
    // internals, public for the meta (-sm tensor) import callback
    ggml_backend_buffer_t import_entry(ggml_context * ctx, ggml_backend_buffer_type_t buft, const char * name);
    void add_export_named(ggml_context * ctx, ggml_backend_buffer_t buf, const char * name);

    // write the manifest and block until killed
    [[noreturn]] void serve();

    struct file_id {
        uint64_t dev, ino, size;
        int64_t  mtime_ns;
    };

    struct entry {
        char     buft[64];
        uint64_t size;
        uint64_t layout;
        uint64_t checksum;
        uint8_t  handle[64];
    };

private:
    bool load_manifest();

    std::string path;
    std::vector<file_id> files;
    std::vector<entry>   entries;
    std::vector<bool>    used;

    bool loaded  = false;
    bool usable  = false;
};
