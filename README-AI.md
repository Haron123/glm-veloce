# glm-veloce: full reference

The detailed companion to [README.md](README.md), written for AI assistants and anyone who wants the details: what the
fork does, measured numbers, every run option, where the code lives and how to benchmark it correctly.

glm-veloce is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), by way of
[castagna-veloce](https://github.com/benpeterson40/castagna-veloce), tuned to run one model as fast as possible on
one machine: **Qwen3.8 Flash-Next (IQ3_XXS) on two AMD Instinct MI50 32 GB cards (gfx906)**. It uses the normal
llama.cpp tools (`llama-server` with its Web UI and OpenAI-compatible API, `llama-bench`, `llama-perplexity`);
everything that makes it faster happens inside the GPU backend and the scheduler.

## Benchmarks

All numbers were measured on the test machine described below, single user, HBM at 1180 MHz.

**Generation speed, layer split (the recommended mode):**

| Test | 120 W cap | 178 W cap |
|---|---|---|
| `llama-bench` tg128, no speculative decoding | 75.3 t/s | 80.3 t/s |
| Server with MTP, "write an i2c master in vhdl", 3000 tokens, greedy | | **104.8 t/s** (87 % of drafts accepted) |
| Server with MTP, same prompt, default sampling | | 101.9 t/s |
| Server with MTP, 12 mixed prompts (code, stories, lists, translation, chat) | | 97.1 t/s overall |
| `llama-bench` at 32K tokens of context, no speculative decoding | 64.6 t/s | |

How much MTP helps depends on how predictable the text is: code and math run at 107-109 t/s, a short story at 84 t/s.

**Prompt processing (prefill), layer split, 120 W:**

| Test | Speed |
|---|---|
| `llama-bench` pp2048, empty context | 931 t/s |
| `llama-bench` pp2048 at 16K context | 831 t/s |
| Server, 12,151-token prompt | 942 t/s |
| Server, 29K-token prompt | 837 t/s |

**Compared with castagna-veloce** (the fork this one started from, commit `3483d46`), same model, same machine,
layer split, 120 W, greedy:

| | castagna-veloce | glm-veloce |
|---|---|---|
| Generation speed (13 open prompts, server) | 33.3 t/s | **71.4 t/s** |
| Graded tasks (math, code run against tests, facts, logic, instructions, 16K/32K needle retrieval) | 54 / 54 | 54 / 54 |
| KL divergence of the output probabilities vs castagna's prefill (16K tokens) | 0.0094 (its own decode path) | 0.0094 |

The speedups do not cost quality: the outputs differ from the baseline only by floating point rounding (different
summation order in fused kernels), at the same level at which the baseline already disagrees with itself between its
prompt and generation code paths. 37 of 67 test outputs were token-for-token identical to the baseline; the rest drift
apart after a few dozen tokens and were all still graded correct.

Tensor split (both GPUs work on every layer) reached 84-85 t/s with `llama-bench` tg128 at a 150 W cap, but it draws
more power, runs hotter and prefills about 30 % slower (~670 t/s on the 12K prompt), so layer split is the default here.

### Test machine

| Part | Details |
|---|---|
| GPUs | 2x AMD Instinct MI50 32 GB (gfx906), PCIe 3.0 x16 each, no bridge between them |
| GPU firmware | Radeon Pro V420 VBIOS on both cards (higher clock and power limits than the stock MI50 VBIOS) |
| HBM clock | 1180 MHz (stock 1000), set through a custom power-play table |
| CPU / RAM | Xeon E5-2680 v4 (14 cores), 30 GB DDR4 |
| Storage | SATA SSD |
| OS | Ubuntu, Linux 7.0 with a two-line patch that enables GPU peer-to-peer copies on this Broadwell host bridge |
| ROCm | 7.2 userspace with the gfx906 build of the core-10.1 compiler (TheRock) |

The VBIOS, the HBM overclock and the kernel patch together are worth roughly 15-20 % of the generation speed. None of
them are needed to run the code.

## What this fork changes

- **Decode kernels written for gfx906.** Fused kernels for the model's hyper-connection projections, the MoE router
  with its top-10 selection, the routed experts in the i-quant formats this quantization uses (iq2_xxs, iq2_s,
  iq3_s, iq4_xs, iq4_nl, q2_0), the shared expert, the gated delta net layers and the sparse attention.
- **Layer split with several stages per GPU.** `-ts 1,1,1,1,1,1` splits the model into six stages that alternate
  between the two cards, which keeps both GPUs warm and is about 5 % faster than one stage per card. Stage hand-offs
  are pushed directly GPU to GPU.
- **Clock keep-alive.** An idle MI50 drops its clocks within a few milliseconds; the scheduler now keeps the waiting
  GPU awake between tokens (+4 %).
- **MTP speculative decoding improvements.** The output head is split across both GPUs, the drafter attends to a
  sliding window, prompt catch-up only stores keys and values, and pooled keys survive the rollback after a rejected
  draft. At 55K tokens of context MTP went from 66 to 75 t/s at 120 W.
- **Prefill.** GEMM tiles for the expert and dense matrices in this model's quantization formats (+10-14 %).
- **Per-layer embedding (PLE) table.** The 27 GB table stays memory-mapped on the CPU; the rows a token needs are
  prefetched, which avoids most disk stalls on a machine with less RAM than the table.
- **Weight keeper** (`tools/mi50/keeper.sh`). A background process holds the 44 GB of weights in VRAM, so a restarted
  server or benchmark maps them in about a second instead of reading them from disk for two minutes.

All optimizations are on by default; most have an environment switch to turn them off for A/B comparisons
(`grep getenv ggml/src/ggml-cuda/*.cu`).

## Build

You need ROCm 7.2 and a compiler that still supports gfx906. This fork is built with TheRock's core-10.1 compiler
installed at `/opt/rocm/core-10.1`.

```
cmake -S . -B build-sub -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang++ -DGGML_SCHED_MAX_COPIES=8 -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_CURL=OFF
cmake --build build-sub -j$(nproc) --target llama-server llama-bench llama-perplexity
```

A clean build takes about four minutes on 28 threads. Add `-DGGML_CCACHE=ON -DCMAKE_HIP_COMPILER_LAUNCHER=ccache`
if you rebuild often. If the build runs out of memory, use fewer jobs (`-j12`).

## Model files

| File | Size | Source |
|---|---|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` + `-00002-of-00002.gguf` | 47 GB + 29 GB | IQ3_XXS quantization of Qwen3.8 Flash-Next; the second part is the per-layer embedding table, which stays in system RAM |
| `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (optional, for MTP) | 2.8 GB | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf) |
| `mmproj-F16.gguf` (optional, for images) | 0.9 GB | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/mmproj-F16.gguf) |

The commands below assume the files live in `~/models/`.

## Run

**Server, layer split with MTP (recommended):**

```
HIP_VISIBLE_DEVICES=0,1 LLAMA_CKPT_LAST_VERIFY=1 ./build-sub/bin/llama-server -m ~/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -md ~/models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-draft-n-max 2 -ngl 999 -fa on -sm layer -ts 1,1,1,1,1,1 -fit off -b 2048 -ub 512 -np 1 -c 131072 --lazy-mode on --jinja --host 0.0.0.0 --port 8080
```

Then open `http://<machine>:8080` for the Web UI, or point any OpenAI-compatible client at it.

- Without MTP: leave out `-md ...` and `--spec-draft-n-max 2`.
- Images: add `--mmproj ~/models/mmproj-F16.gguf` (0.9 GB more VRAM).
- Shorter answers with less thinking: add `--chat-template-kwargs '{"reasoning_effort":"low"}'`.
- `--spec-draft-n-max 2` is the measured sweet spot; three drafts per step is slower.

**Context size and VRAM.** Six stages need more VRAM than two. With MTP:

| `-c` | Result |
|---|---|
| 65536 | fits with pipelined prefill, the fastest setting for long prompts (measured with `-ts 1.28,1,1.28,1,1.28,1`) |
| 131072 | fits, but the server drops pipeline parallelism (log: `retrying without pipeline parallelism`): generation speed is the same, long prompts prefill more slowly because only one GPU works at a time |
| 204800 | did not fit with six stages; the two-stage split `-ts 1.3,1` runs it at about 5 % slower generation |

**Server, tensor split:**

```
HIP_VISIBLE_DEVICES=0,1 LLAMA_CKPT_LAST_VERIFY=1 ./build-sub/bin/llama-server -m ~/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -md ~/models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-draft-n-max 2 -ngl 999 -fa on -sm tensor -b 2048 -ub 512 -np 1 -c 100000 --lazy-mode on --jinja --host 0.0.0.0 --port 8080
```

Do not pass `-ts` in tensor mode (an uneven split crashes one of the kernels).

**Benchmark:**

```
HIP_VISIBLE_DEVICES=0,1 ./build-sub/bin/llama-bench -m ~/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -ngl 999 -fa 1 -sm layer -ts 1/1/1/1/1/1 -p 0 -n 128 -r 4
```

`llama-bench` separates the split values with `/`. The first repetition is always slower; use the later ones. Run one
test per process: a second test in the same process reads about a third lower.

**Fast restarts with the weight keeper:**

```
SM=layer tools/mi50/keeper.sh start -ts 1,1,1,1,1,1
LLAMA_WEIGHT_KEEPER=/dev/shm/glmv-keeper LLAMA_LAZY_PREFETCH=0 HIP_VISIBLE_DEVICES=0,1 ./build-sub/bin/llama-bench ... -sm layer -ts 1/1/1/1/1/1 ...
tools/mi50/keeper.sh stop
```

The keeper occupies the VRAM, so stop it before starting a normal server. Programs that use it must use the same build
and the same split.

## Power and temperature

The MI50 is passively cooled. At a 178 W cap, long generations can push the GPU junction temperature to 100 °C within
a few minutes unless the cards get a lot of air; 120 W is the sustainable setting on the test machine and costs about
6 % of the generation speed. Set the cap with `sudo rocm-smi --setpoweroverdrive 120` (it resets on reboot) and keep
an eye on `rocm-smi`.

## Limitations

- Only gfx906 (MI50, MI60, Radeon VII class cards) is tuned and tested. Other GPUs may be slower than upstream
  llama.cpp, or not work at all.
- The kernels are tuned for this one model and its quantization mix. Other models fall back to the regular llama.cpp
  code paths.
- The per-layer embedding table (27 GB) does not fit in 30 GB of RAM together with everything else. Text the model has
  not seen recently runs a few percent slower while rows are read from the SSD. More RAM or an NVMe drive fixes that.

## Where the code lives

| Area | Files |
|---|---|
| Fused decode kernels (hyper-connections, shared expert, router, GDN) | `ggml/src/ggml-cuda/hc-dec.cu`, `moe-f16.cu`, `shexp-dec.cu`, `topk-moe.cu`, `hc-persist.cu` |
| Routed-expert kernels (i-quants) | `ggml/src/ggml-cuda/moe-vec.cu`, `moe-iq*.cu`, `moe-vecq*.cu` |
| Quantized matvecs and prefill GEMM tiles | `gcn-q6k-dec.cu`, `gcn-kq-matvec.cu`, `mmvq.cu`, `gcn-q8-gemm.cu` |
| Sparse attention (QSA) | `getrows.cu` (pooled-key scorer), `fattn-sparse-gcn.cu`, `top-k.cu`; host side in `src/llama-memory-hybrid-idx.cpp` |
| Fusion matcher, graph capture, P2P hand-offs | `ggml/src/ggml-cuda/ggml-cuda.cu`, `allreduce.cu` |
| Scheduler (stages, keep-alive, events) | `ggml/src/ggml-backend.cpp`, `ggml-backend-meta.cpp` |
| Model graph | `src/models/qwen4exp.cpp` |
| MTP drafting | `common/speculative.cpp`, `tools/server/server-context.cpp` |
| Weight keeper | `src/llama-weight-keeper.cpp`, `tools/mi50/keeper.sh` |
| Kernel tests | `tests/test-backend-ops.cpp` (e.g. `-o MUL_MAT_MULTI`, `-o HC_UP_PRE`, `-o SHEXP_DEC`, `-o QSA_GATHER_SCORE`) |
| Benchmark scripts | `tools/mi50/` |

## How to measure changes

- Speed: A/B both sides in one build through the environment switch, as interleaved pairs (A B A B ...), at the same
  power cap and with cards that start cool; use the steady repetitions, not the first. Process-to-process noise is
  about 1-1.5 %.
- Correctness: `test-backend-ops` for the touched op, then greedy output identity on a fixed prompt with
  `llama-server` (temperature 0). A passing test does not prove your kernel ran: check with its switch turned off.
- Quality, for anything that changes numerics: `llama-perplexity --kl-divergence-base` / `--kl-divergence` against
  the previous build. Reference: the 1-token decode path differs from the prefill path by KLD ~0.009 on its own, so
  changes at that level are rounding; re-quantizing weight groups to 4 bit measured 0.014-0.13.
- `llama-bench` replays the same tokens every run, so the per-layer embedding rows are cached and its numbers are a
  little optimistic for new text.

## License

MIT, see [LICENSE](LICENSE).
