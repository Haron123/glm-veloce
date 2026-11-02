#!/usr/bin/env python3
"""tg-vs-context-depth curve for MI50 llama-servers.

Sends growing prefixes of the PGE prompt with a fixed greedy generation and
records decode tok/s at each depth. The server caches the common prefix, so
each step prefills only the delta. The shape of the curve says where the
context cost lives: a step at n_kv = top_k + r - 1 (2051) means the sparse
indexer path; smooth linear growth means KV/FA reads.

Usage: bench_tgcurve.py --port 8080 --label tgcurve [--gen 64]
Saves runs/<label>.json
"""

import argparse
import json
import os
import urllib.request

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pge-prompt.txt")
DEPTHS = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.90]


def http_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--label", default="tgcurve")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--gen", type=int, default=64)
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/completion"
    with open(PROMPT_FILE) as f:
        prompt = f.read()
    tok_per_char = 98601.0 / len(prompt)

    results = []
    for frac in DEPTHS:
        cut = int(len(prompt) * frac)
        payload = {
            "prompt": prompt[:cut],
            "max_tokens": args.gen,
            "temperature": 0.0,
            "cache_prompt": True,
            "ignore_eos": True,
        }
        body = http_json(url, payload, timeout=3600)
        t = body.get("timings", {})
        tg = t["predicted_n"] / (t["predicted_ms"] / 1000.0) if t.get("predicted_ms") else None
        depth = int(cut * tok_per_char)
        results.append({"depth_tokens_est": depth, "tg_tok_s": tg,
                        "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                        "prompt_n_delta": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms")})
        print(f"depth~{depth:6d} tok | tg {tg:6.2f} tok/s ({t.get('predicted_n')} tok) | prefill delta {t.get('prompt_n')} in {t.get('prompt_ms', 0)/1000.0:.1f}s")

    os.makedirs(args.runs_dir, exist_ok=True)
    out = os.path.join(args.runs_dir, f"{args.label}.json")
    with open(out, "w") as f:
        json.dump({"label": args.label, "results": results}, f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
