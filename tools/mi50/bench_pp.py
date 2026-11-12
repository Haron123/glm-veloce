#!/usr/bin/env python3
# Prompt-processing (pp) rate on a running llama-server: a ~N-token slice of tools/mi50/pge-prompt.txt (C++ source),
# a unique first line per run so nothing is served from the prompt cache, 1 generated token. Reports the server's own
# prompt timings (prompt_n / prompt_ms); the first run is a discarded warmup.
#   python3 tools/mi50/bench_pp.py [--port 8080] [--chars 10000] [--runs 4]
import argparse
import json
import os
import statistics
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--chars", type=int, default=10000, help="prompt length in characters (~3.4 chars per token)")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--out", default=None, help="optional json file for the results")
    args = ap.parse_args()

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pge-prompt.txt"), encoding="utf-8").read()
    body = src[:args.chars]
    res = []
    for i in range(args.runs + 1):
        prompt = f"// run {time.time_ns()} {i}\n" + body
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}/completion",
                                     data=json.dumps({"prompt": prompt, "n_predict": 1, "temperature": 0,
                                                      "cache_prompt": False}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            t = json.loads(r.read())["timings"]
        pps = t["prompt_n"] / (t["prompt_ms"] / 1000.0)
        tag = "warmup" if i == 0 else f"run {i}"
        print(f"[{tag}] prompt_n={t['prompt_n']} prompt_ms={t['prompt_ms']:.1f} pp={pps:.1f} tok/s", flush=True)
        if i > 0:
            res.append(pps)
    print(f"pp mean {statistics.mean(res):.1f} tok/s, median {statistics.median(res):.1f} (n={len(res)}, "
          f"{t['prompt_n']} tokens)")
    if args.out:
        json.dump({"pp": res, "prompt_n": t["prompt_n"]}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
