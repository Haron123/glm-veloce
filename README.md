# glm-veloce
Omega-Vibecoded fork of [castagna-veloce](https://github.com/benpeterson40/castagna-veloce)

Deepseek, GLM, Qwen and Opus did this as a joint group project for weeks. Its better than castagne-veloce in layer split and tensor parallel on 2xMI50 32GB

Output quality was barely tested (Opus 5.5 said its OK :D?), if anyone wants to run a testsuite i can update the README.md with it.

### Expected speed (2xMI50 32GB) on Qwen Flash Next IQ3_XSS with MTP
Layer: ~700-1000pp/s | ~50-90tg/s depending on context
Tensor: ~300-500pp/s | ~60-110tg/s depending on context

## Build
My build command: 
```
cmake -S . -B build-sub -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang++ -DGGML_SCHED_MAX_COPIES=8 -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_CURL=OFF
cmake --build build-sub -j$(nproc) --target llama-server
```

## Run
Layer-Split in this configuration fits the model, 256k context, the mmproj and the drafter:
```
HIP_VISIBLE_DEVICES=0,1 LLAMA_CKPT_LAST_VERIFY=1 LLAMA_MTP_DRAFT_VOCAB=131072 ./build-sub/bin/llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.gguf --mmproj mmproj-F16.gguf -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-draft-n-max 2 -ngl 999 -fa on -fit off -sm layer -ts 1.30,1 -b 2048 -ub 320 -np 1 -c 262144 --lazy
-mode on --jinja --host 0.0.0.0 --port 8080 -cram 4096
```
Changing -ub to 512 can increase prefill speed but won't fit in 64GB VRAM

Tensor-Split
```
HIP_VISIBLE_DEVICES=0,1 LLAMA_CKPT_LAST_VERIFY=1 ./build-sub/bin/llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.gguf -ngl 999 -fa on -sm tensor -b 2048 -ub 512 -np 1 -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-draft-n-max 2 -c 100000 --lazy-mode on --host 0.0.0.0 --jinja --port 8080
```

Document written by AI for AI, theres more details there i assume: [README-AI.md](README-AI.md).
