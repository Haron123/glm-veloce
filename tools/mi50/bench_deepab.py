#!/usr/bin/env python3
"""Deep-context A/B harness for MI50 llama-servers.

Sends a deep prefix of the PGE prompt (98,601 tokens) with a greedy fixed-length
generation, saving the OUTPUT TEXT plus timings, so two builds (or two env
settings) can be diffed for quality at long context. The server must run with
cache ON and -c >= 102400.

Usage:
  bench_deepab.py --port 8080 --label <name> [--frac 0.90] [--gen 192]
Saves runs/<label>/output.txt and runs/<label>/meta.json
"""

import argparse
import json
import os
import time
import urllib.request

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pge-prompt.txt")


def http_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--frac", type=float, default=0.90)
    ap.add_argument("--gen", type=int, default=192)
    args = ap.parse_args()

    url_base = f"http://127.0.0.1:{args.port}"

    with open(PROMPT_FILE, "r") as f:
        prompt = f.read()
    cut = int(len(prompt) * args.frac)
    prefix = prompt[:cut]
    tok_per_char = 98601.0 / len(prompt)

    payload = {
        "prompt": prefix,
        "max_tokens": args.gen,
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
    content = body.get("content", "")

    os.makedirs(args.runs_dir, exist_ok=True)
    out_dir = os.path.join(args.runs_dir, args.label)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "output.txt"), "w") as f:
        f.write(content)
    meta = {
        "label": args.label,
        "frac": args.frac,
        "depth_tokens_est": int(cut * tok_per_char),
        "gen_tokens": pnv,
        "tg_tok_s": tg,
        "prompt_n_delta": pn,
        "prompt_ms": pt,
        "wall_s": wall,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    print(f"label={args.label} depth~{meta['depth_tokens_est']} tg={tg:.2f} tok/s "
          f"({pnv} tokens, cached prefill delta {pn} in {pt/1000.0:.2f}s)")
    print(f"saved {out_dir}/output.txt")


if __name__ == "__main__":
    main()
