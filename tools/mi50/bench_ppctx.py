#!/usr/bin/env python3
"""pp-vs-context-depth benchmark for MI50 llama-servers.

Measures how marginal prompt-processing speed degrades as the KV context
fills up. Protocol:

- Prompt source: the PGE prompt (tools/mi50/pge-prompt.txt, 98,601 tokens).
- Server MUST run with cache ON (no --no-cache-prompt) and -c >= 102400.
- Sends a sequence of requests with GROWING prefixes of the prompt. The
  server caches the common prefix, so each request prefills only the delta;
  marginal pp at depth d = delta_tokens / prompt_ms from server timings.
- Ends with a fixed-length generation off the deepest prefix to measure tg
  at long context (predicted_n / predicted_ms).

Usage:
  bench_ppctx.py --port 8080 --label ppctx --runs-dir runs
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pge-prompt.txt")

# fractions of the file (by bytes) -> approximate token depths
DEPTH_STEPS = [0.02, 0.10, 0.20, 0.40, 0.65, 0.90]


def http_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body, time.time() - t0


def wait_health(url_base, deadline_s):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            with urllib.request.urlopen(url_base + "/health", timeout=5) as r:
                if json.loads(r.read().decode("utf-8")).get("status") == "ok":
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("server did not become healthy within deadline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--label", default="ppctx")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--gen-tokens", type=int, default=192)
    ap.add_argument("--health-deadline", type=int, default=1800)
    args = ap.parse_args()

    url_base = f"http://127.0.0.1:{args.port}"
    wait_health(url_base, args.health_deadline)

    with open(PROMPT_FILE, "r") as f:
        prompt = f.read()

    n_chars = len(prompt)
    # token estimate from the known ratio (98601 tokens / 273877 bytes)
    tok_per_char = 98601.0 / n_chars

    results = []
    prev_tokens = 0

    for frac in DEPTH_STEPS:
        cut = int(n_chars * frac)
        prefix = prompt[:cut]
        est_tokens = int(cut * tok_per_char)
        payload = {
            "prompt": prefix,
            "max_tokens": 1,
            "temperature": 0.0,
            "cache_prompt": True,
        }
        body, wall = http_json(url_base + "/completion", payload, timeout=3600)
        pt = body.get("timings", {}).get("prompt_ms", None)
        pn = body.get("timings", {}).get("prompt_n", None)
        delta_n = pn if pn is not None else est_tokens - prev_tokens
        depth = prev_tokens + delta_n / 2 if pn is not None else (prev_tokens + est_tokens) / 2
        pp = delta_n / (pt / 1000.0) if pt else None
        results.append({
            "frac": frac,
            "cut_bytes": cut,
            "est_tokens_total": est_tokens,
            "prompt_n_delta": pn,
            "prompt_ms": pt,
            "marginal_pp_tok_s": pp,
            "depth_tokens_est": int(depth),
            "wall_s": wall,
        })
        print(f"depth~{int(depth):6d} tok | delta {delta_n:6d} tok in {pt/1000.0:7.2f}s -> marginal pp {pp:7.1f} tok/s")
        prev_tokens = est_tokens

    # tg at the deepest context
    cut = int(n_chars * DEPTH_STEPS[-1])
    prefix = prompt[:cut]
    payload = {
        "prompt": prefix,
        "max_tokens": args.gen_tokens,
        "temperature": 0.0,
        "cache_prompt": True,
        "ignore_eos": True,
    }
    body, wall = http_json(url_base + "/completion", payload, timeout=3600)
    pt = body.get("timings", {}).get("prompt_ms")
    pn = body.get("timings", {}).get("prompt_n")
    pm = body.get("timings", {}).get("predicted_ms")
    pnv = body.get("timings", {}).get("predicted_n")
    tg = pnv / (pm / 1000.0) if pm else None
    print(f"tg at depth ~{int(cut * tok_per_char)}: {tg:.2f} tok/s ({pnv} tokens; cached prefill delta {pn} in {pt/1000.0:.2f}s)")
    results.append({
        "phase": "tg_at_depth",
        "depth_tokens_est": int(cut * tok_per_char),
        "tg_tok_s": tg,
        "predicted_n": pnv,
        "predicted_ms": pm,
        "prompt_n_delta": pn,
        "prompt_ms": pt,
    })

    os.makedirs(args.runs_dir, exist_ok=True)
    out = os.path.join(args.runs_dir, f"{args.label}.json")
    with open(out, "w") as f:
        json.dump({"label": args.label, "results": results}, f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
