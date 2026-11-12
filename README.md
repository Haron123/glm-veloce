# glm-veloce

Vibecoded fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (via
[castagna-veloce](https://github.com/benpeterson40/castagna-veloce)), specifically for Qwen3.8 Flash-Next
(IQ3_XXS) on 2x AMD Instinct MI50 32 GB (gfx906).

Agents and contributors: read [current.md](current.md) first. It is the only project document.

## Build

ROCm 7.2 userspace with the core-10.1 (TheRock gfx906) compiler at `/opt/rocm/core-10.1`.

```
cmake -S . -B build-sub -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang++ -DGGML_CCACHE=ON -DCMAKE_HIP_COMPILER_LAUNCHER=ccache -DGGML_SCHED_MAX_COPIES=8 -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_CURL=OFF
cmake --build build-sub -j28 --target llama-server llama-bench test-backend-ops llama-perplexity
```

## Run

Server, tensor split (raw decoding):

```
HIP_VISIBLE_DEVICES=0,1 ./build-sub/bin/llama-server -m ~/models/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -ngl 999 -fa on -sm tensor -b 2048 -ub 512 -np 1 -c 100000 --jinja --host 0.0.0.0 --port 8080
```

Daily use with the MTP drafter: add `-md ~/models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-draft-n-max 2`.
Do not pass `-ts` in tensor mode.

Weight keeper (keeps the 44 GB of weights in VRAM so restarts take seconds instead of minutes):

```
tools/mi50/keeper.sh start
LLAMA_WEIGHT_KEEPER=/dev/shm/glmv-keeper LLAMA_LAZY_PREFETCH=0 HIP_VISIBLE_DEVICES=0,1 ./build-sub/bin/llama-bench -m ~/models/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -ngl 999 -fa 1 -sm tensor -p 0 -n 128 -r 4
tools/mi50/keeper.sh stop
```

## Benchmarks

2x MI50, 150 W cap, HBM 1180 MHz, tensor split, no speculative decoding (2026-11-10):

| Test | Result |
|---|---|
| llama-bench tg128 (steady, reps 2-4) | 84.0-84.8 t/s |
| server, "write an i2c master in vhdl", greedy | 83.4-83.6 t/s |
| server prefill, 12151-token prompt | 669-673 t/s (layer split: 887-950) |
| novel tokens with the PLE table not cached in RAM | ~77 t/s |
| server with MTP (daily use) | ~85-94 t/s |

History, methods, quality gates and open work: [current.md](current.md).

## License

MIT, see [LICENSE](LICENSE).
