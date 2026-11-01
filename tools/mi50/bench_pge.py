#!/usr/bin/env python3
"""PGE-prompt benchmark for MI50 llama-servers.

Protocol (user-mandated):
- Prompt = full olcPixelGameEngine.h v2.30 source (tools/mi50/pge-prompt.txt).
- Prompt caching stays ON (server is never started with --no-cache-prompt;
  all requests send "cache_prompt": true).
- pp: repeated near-full prefills. Each pp request mutates one comment line at
  the very top of the prompt, so the common cached prefix is ~1-2 tokens and
  the server prefills essentially the whole prompt every time. Rate comes from
  the server's own timings (prompt_n / prompt_ms).
- tg: repeated generations off the fully cached prompt (identical prompt text
  every request -> full cache hit, pure decode). Rate from predicted_n /
  predicted_ms. Fixed-length generation via ignore_eos.
- Report: mean, 1% low (p1), 1% high (p99), nearest-rank percentiles.

Usage:
  bench_pge.py --port 8080 --n-pp 12 --n-tg 30 --gen-tokens 192 \
      --label rcm101 --runs-dir runs
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pge-prompt.txt")


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


def percentile_nearest(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    import math
    idx = max(0, min(len(sorted_vals) - 1, math.ceil(p / 100.0 * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def stats(vals):
    s = sorted(vals)
    mean = sum(s) / len(s)
    return {
        "n": len(s),
        "mean": mean,
        "p1_low": percentile_nearest(s, 1),
        "p99_high": percentile_nearest(s, 99),
        "median": percentile_nearest(s, 50),
        "min": s[0],
        "max": s[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n-pp", type=int, default=12)
    ap.add_argument("--n-tg", type=int, default=30)
    ap.add_argument("--gen-tokens", type=int, default=192)
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    # pp variants: one differing comment line at the very top
    def variant(i):
        return f"// olc pge bench prefill variant {i:04d}\n" + base_prompt

    base = f"http://{args.host}:{args.port}"
    out = {"label": args.label, "port": args.port,
           "prompt_bytes": len(base_prompt), "n_pp": args.n_pp, "n_tg": args.n_tg,
           "gen_tokens": args.gen_tokens, "pp": [], "tg": []}

    pp_reqs = [variant(i) for i in range(args.n_pp)]
    tg_prompt = variant(args.n_pp - 1)  # fully cached after the last pp run

    print(f"[{args.label}] waiting for health on :{args.port} ...", flush=True)
    print(f"[{args.label}] healthy after {wait_health(base, 1800):.1f}s", flush=True)

    # ---- pp samples (near-full prefill each run, cache ON) ----
    for i, p in enumerate(pp_reqs):
        payload = {
            "prompt": p,
            "n_predict": 1,
            "temperature": 0.0,
            "cache_prompt": True,
            "stream": False,
        }
        try:
            body, wall = http_json(base + "/completion", payload, args.timeout)
        except urllib.error.HTTPError as e:
            print(f"[{args.label}] pp run {i}: HTTP {e.code}: {e.read()[:300]}", flush=True)
            raise
        t = body.get("timings", {})
        pn, pms = t.get("prompt_n"), t.get("prompt_ms")
        rec = {"run": i, "prompt_n": pn, "prompt_ms": pms, "wall_s": round(wall, 2),
               "pp": (pn / pms * 1000.0) if (pn and pms) else None,
               "cache_tokens": body.get("cache_tokens"),
               "tokens_predicted": body.get("tokens_predicted"),
               "tokens_evaluated": body.get("tokens_evaluated")}
        out["pp"].append(rec)
        print(f"[{args.label}] pp {i+1}/{args.n_pp}: prompt_n={pn} prompt_ms={pms} "
              f"pp={rec['pp']:.1f} tok/s (wall {wall:.1f}s)", flush=True)

    n_prompt_tokens = out["pp"][0]["prompt_n"]

    # ---- tg samples (identical cached prompt, fixed-length decode) ----
    for i in range(args.n_tg):
        payload = {
            "prompt": tg_prompt,
            "n_predict": args.gen_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "cache_prompt": True,
            "stream": False,
        }
        try:
            body, wall = http_json(base + "/completion", payload, args.timeout)
        except urllib.error.HTTPError as e:
            print(f"[{args.label}] tg run {i}: HTTP {e.code}: {e.read()[:300]}", flush=True)
            raise
        t = body.get("timings", {})
        gn, gms = t.get("predicted_n"), t.get("predicted_ms")
        rec = {"run": i, "predicted_n": gn, "predicted_ms": gms, "wall_s": round(wall, 2),
               "tg": (gn / gms * 1000.0) if (gn and gms) else None,
               "prompt_ms": t.get("prompt_ms"), "prompt_n": t.get("prompt_n")}
        out["tg"].append(rec)
        print(f"[{args.label}] tg {i+1}/{args.n_tg}: predicted_n={gn} ms={gms} "
              f"tg={rec['tg']:.2f} tok/s (prompt_ms={t.get('prompt_ms')})", flush=True)

    out["prompt_n_first"] = n_prompt_tokens
    out["pp_stats"] = stats([r["pp"] for r in out["pp"] if r["pp"]])
    out["tg_stats"] = stats([r["tg"] for r in out["tg"] if r["tg"]])

    print("\n==== RESULTS:", args.label, "====", flush=True)
    print("prompt tokens (first full prefill):", n_prompt_tokens, flush=True)
    for k in ("pp_stats", "tg_stats"):
        st = out[k]
        print(f"{k}: n={st['n']} mean={st['mean']:.2f} p1_low={st['p1_low']:.2f} "
              f"p99_high={st['p99_high']:.2f} median={st['median']:.2f} "
              f"min={st['min']:.2f} max={st['max']:.2f}", flush=True)

    if args.runs_dir:
        os.makedirs(args.runs_dir, exist_ok=True)
        path = os.path.join(args.runs_dir, f"pge-{args.label}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print("saved:", path, flush=True)


if __name__ == "__main__":
    main()
