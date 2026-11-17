# glm-veloce

A [llama.cpp](https://github.com/ggml-org/llama.cpp) fork that runs **Qwen3.8 Flash-Next** fast on two
**AMD Instinct MI50** (32 GB, gfx906) cards.

- **~105 t/s** writing code with MTP speculative decoding, 75-80 t/s without
- **~940 t/s** prompt processing
- **2.1x faster** than [castagna-veloce](https://github.com/benpeterson40/castagna-veloce), which it is based on, with
  the same output quality

## Build

Needs ROCm 7.2 and a gfx906-capable compiler (TheRock core-10.1 at `/opt/rocm/core-10.1`).

```
cmake -S . -B build-sub -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang++ -DGGML_SCHED_MAX_COPIES=8 -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_CURL=OFF
cmake --build build-sub -j$(nproc) --target llama-server
```

## Run

```
HIP_VISIBLE_DEVICES=0,1 LLAMA_CKPT_LAST_VERIFY=1 ./build-sub/bin/llama-server -m ~/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -md ~/models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-draft-n-max 2 -ngl 999 -fa on -sm layer -ts 1,1,1,1,1,1 -fit off -b 2048 -ub 512 -np 1 -c 131072 --lazy-mode on --jinja --host 0.0.0.0 --port 8080
```

Open `http://<machine>:8080`. The MTP drafter is
[here](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf).

The MI50s are passively cooled: give them plenty of air, or cap them at 120 W (`sudo rocm-smi --setpoweroverdrive 120`).

Benchmarks, all options and how it works: [README-AI.md](README-AI.md).

MIT license.
