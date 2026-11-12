#!/usr/bin/env python3
# Marginal prompt processing at context depth on a running llama-server (cache ON, CACHE=1 in serve.sh):
# for each depth D, the first D tokens of tools/mi50/pge-prompt.txt are prefilled (cached), then the next ~CHUNK
# tokens are sent as an extension of the same prompt, and the server's own timings of that delta give pp at depth D.
#   python3 tools/mi50/bench_ppdepth.py [--port 8080] [--depths 0,8192,16384,24576] [--chunk 2048]
import argparse
import json
import os
import urllib.request

CHARS_PER_TOK = 3.67  # pge-prompt.txt (C++ source)


def post(port, prompt):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps({"prompt": prompt, "n_predict": 1, "temperature": 0,
                                                  "cache_prompt": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())["timings"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--depths", default="0,8192,16384,24576")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pge-prompt.txt"), encoding="utf-8").read()
    res = []
    for i, d in enumerate(int(x) for x in args.depths.split(",")):
        head = f"// depth run {os.getpid()} {i}\n"  # a fresh prefix per depth: nothing reused across depths
        a = int(d*CHARS_PER_TOK)
        b = int((d + args.chunk)*CHARS_PER_TOK)
        if d > 0:
            post(args.port, head + src[:a])
        t = post(args.port, head + src[:b])
        pps = t["prompt_n"] / (t["prompt_ms"] / 1000.0)
        print(f"depth {d:6d}: {t['prompt_n']} new tokens in {t['prompt_ms']:.0f} ms -> {pps:.0f} tok/s", flush=True)
        res.append({"depth": d, "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"], "pp": pps})
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
