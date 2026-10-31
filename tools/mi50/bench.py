#!/usr/bin/env python3
"""Benchmark harness for llama-server: tg (decode) and TTFT (prompt processing).

Sends a fixed set of varied prompts to a running llama-server and measures, per
prompt: time to first token (TTFT, the prompt-processing cost of that prompt)
and decode speed (tokens per second after the first token). Greedy decoding,
prompt cache off, so runs are comparable.

Usage: python3 tools/mi50/bench.py <port> --out runs/<name> [--max-tokens 256]

Not llama-bench: this measures the server the way a user sees it, over the same
fixed prompts every time, and saves every response for quality comparison.
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

PROMPTS = [
    ("code-gen", "Write a Python function that merges two sorted lists into one sorted list, with tests."),
    ("code-gen", "Write a C++ class for a fixed-size thread-safe ring buffer."),
    ("code-explain", "Explain what this regex does: ^(?=.*[a-z])(?=.*\\d)[a-zA-Z\\d]{8,}$"),
    ("code-explain", "What is the difference between a mutex and a semaphore? When would you use each?"),
    ("list", "List 10 practical tips for writing maintainable code."),
    ("story", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
    ("reasoning", "If a train leaves at 3pm traveling 80 km/h and another leaves at 4pm traveling 120 km/h on the same track, when does the second catch up? Show the steps."),
    ("math", "Compute the derivative of f(x) = x^3 * sin(x) and simplify."),
    ("json", "Return a JSON object describing a book: title, author, year, genres (array), and rating. Use a made-up book."),
    ("translation", "Translate into French: 'The quick brown fox jumps over the lazy dog.' Then explain the idiom."),
    ("summarization", "Summarize the key ideas of the scientific method in a few sentences."),
    ("chat", "What are the trade-offs between buying a used server GPU and a new consumer GPU for machine learning?"),
]


def post(port, path, payload, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def bench_one(port, category, prompt, max_tokens):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    text_parts = []
    finish = None
    usage = {}
    with post(port, "/v1/chat/completions", payload) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tokens += 1
                text_parts.append(content)
            if choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
    t_end = time.perf_counter()
    if ttft is None:
        return None
    decode_time = t_end - t0 - ttft
    tg = (tokens - 1) / decode_time if tokens > 1 and decode_time > 0 else 0.0
    return {
        "category": category,
        "prompt": prompt,
        "ttft_s": ttft,
        "tokens": tokens,
        "tg_tok_s": tg,
        "total_s": t_end - t0,
        "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "output": "".join(text_parts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--out", required=True, help="output directory for this run")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=1, help="times to run the whole set (first is warmup if --warmup)")
    ap.add_argument("--warmup", action="store_true", help="run the set once first and discard")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    def run_set(tag):
        results = []
        for category, prompt in PROMPTS:
            r = bench_one(args.port, category, prompt, args.max_tokens)
            if r is None:
                print(f"FAILED: {category}: {prompt[:50]}", file=sys.stderr)
                sys.exit(1)
            r["run"] = tag
            results.append(r)
            print(f"[{tag}] {category:14s} ttft={r['ttft_s']*1000:7.0f} ms  tg={r['tg_tok_s']:6.2f} tok/s  ({r['tokens']} tokens)")
        return results

    if args.warmup:
        print("warmup run (discarded)")
        run_set("warmup")

    all_results = []
    for i in range(args.repeat):
        all_results.extend(run_set(f"r{i+1}"))

    with open(os.path.join(args.out, "outputs.jsonl"), "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    tgs = [r["tg_tok_s"] for r in all_results if r["tokens"] > 1]
    ttfts = [r["ttft_s"] for r in all_results]
    summary = {
        "n": len(all_results),
        "tg_mean": statistics.mean(tgs),
        "tg_median": statistics.median(tgs),
        "ttft_mean_s": statistics.mean(ttfts),
        "ttft_median_s": statistics.median(ttfts),
        "tg_by_category": {},
        "ttft_by_category": {},
    }
    for category, _ in PROMPTS:
        cat_tg = [r["tg_tok_s"] for r in all_results if r["category"] == category and r["tokens"] > 1]
        cat_ttft = [r["ttft_s"] for r in all_results if r["category"] == category]
        if cat_tg:
            summary["tg_by_category"][category] = statistics.mean(cat_tg)
            summary["ttft_by_category"][category] = statistics.mean(cat_ttft)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print()
    print(f"== SUMMARY ({summary['n']} runs) ==")
    print(f"tg  (decode):   mean {summary['tg_mean']:.2f} tok/s   median {summary['tg_median']:.2f} tok/s")
    print(f"ttft (prompt):  mean {summary['ttft_mean_s']*1000:.0f} ms       median {summary['ttft_median_s']*1000:.0f} ms")
    print(f"saved: {args.out}/outputs.jsonl, {args.out}/summary.json")


if __name__ == "__main__":
    main()

# --- honest first-see mode helpers (appended) ---
# The n-gram speculative index persists across requests (shared across
# sequences, resets only at 25% occupancy). A warmup run with the SAME prompts
# lets the measured run draft the warmup's outputs, inflating tg. For honest
# numbers: fresh server, warm up with UNIQUE prompts, then measure each
# benchmark prompt the first time it is seen (no --warmup).
UNIQUE_WARMUP_PROMPTS = [
    "Explain how a differential gear works in a car.",
    "Write a haiku about autumn rain.",
    "What is the capital of Australia and why is it not Sydney?",
]
