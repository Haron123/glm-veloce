#!/usr/bin/env python3
"""Task-quality A/B for rounding-class kernel changes.

Runs a suite of verifiable tasks (math, code, facts, logic, instruction
following) against a llama-server and records the raw answers, so two builds
can be compared for task CORRECTNESS (not just wording).

Usage: bench_tasks.py --port 8080 --out runs/<name>/tasks.jsonl
"""

import argparse
import json
import os
import urllib.request

TASKS = [
    ("math", "What is 17 * 23? Answer with just the number.", ["391"]),
    ("math", "Solve for x: 3x + 7 = 22. Answer with just the number.", ["5"]),
    ("math", "What is the derivative of x^2? Answer with just the expression.", ["2x", "2*x", "2 x"]),
    ("code", "Write a Python one-liner that reverses the string s. Answer with only the code.", ["s[::-1]", "s[::-1] "]),
    ("code", "In Python, what does mylist.append(x) return? One sentence.", ["none", "in-place", "modifies"]),
    ("code", "Write a Python function is_even(n) returning True if n is even. Only the code.", ["n % 2 == 0", "n%2==0"]),
    ("fact", "What is the capital of Australia? One word.", ["canberra"]),
    ("fact", "In what year did the Berlin Wall fall? Just the year.", ["1989"]),
    ("fact", "What chemical element has the symbol Fe? One word.", ["iron"]),
    ("logic", "If all bloops are razzles and all razzles are lazzles, are all bloops definitely lazzles? Answer yes or no.", ["yes"]),
    ("count", "How many days are in February in a leap year? Just the number.", ["29"]),
    ("instruction", "List exactly three primary colors, comma-separated, nothing else.", ["red", "blue", "yellow"]),
    ("translation", "Translate to French: 'the cat is black'. Only the translation.", ["le chat est noir", "chat est noir"]),
    ("json", "Return only valid JSON: {\"a\": 1, \"b\": [2, 3]}. Nothing else.", ["{\"a\": 1", '"a": 1']),
    ("reasoning", "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much is the ball? Just the number with currency.", ["0.05", "5 cents", "five cents", "$0.05"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/completion"
    results = []
    n_correct = 0
    for cat, prompt, accepts in TASKS:
        payload = {"prompt": prompt, "max_tokens": 64, "temperature": 0.0, "cache_prompt": False}
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        body = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
        text = body.get("response", "") or body.get("content", "")
        low = text.lower().strip()
        correct = any(a.lower() in low for a in accepts)
        n_correct += correct
        results.append({"category": cat, "prompt": prompt, "response": text, "correct": correct})
        print(f"[{'OK ' if correct else 'FAIL'}] {cat:12s} {prompt[:50]:52s} -> {text[:60]!r}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nTask score: {n_correct}/{len(TASKS)}")


if __name__ == "__main__":
    main()
