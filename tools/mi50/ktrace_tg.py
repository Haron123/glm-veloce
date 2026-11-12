#!/usr/bin/env python3
# Per-token decode breakdown from a rocprofv3 kernel trace (CSV):
#   rocprofv3 --kernel-trace --output-format csv -d DIR -o tg -- llama-bench ... -p 0 -n 64 -r 1
#   python3 tools/mi50/ktrace_tg.py DIR/tg_kernel_trace.csv [--tokens 48] [--top 40]
# Tokens are cut at host gaps (no kernel on any GPU for --gap-us). Reports per token: kernel count, the union
# of busy time over both GPUs, idle time, and the kernels ranked by total time with their per-call median.
import argparse
import csv
import collections
import re
import statistics


def short(name):
    name = re.sub(r"\(.*", "", name)
    name = re.sub(r"^void ", "", name)
    return name[:110]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--tokens", type=int, default=48, help="analyse the last N tokens")
    ap.add_argument("--gap-us", type=float, default=150.0, help="idle gap that separates two tokens")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--full-names", action="store_true")
    ap.add_argument("--dump", type=int, default=0, help="print the first N kernels of the last token in order")
    args = ap.parse_args()

    ks = []
    with open(args.trace) as f:
        for r in csv.DictReader(f):
            ks.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Agent_Id"], r["Kernel_Name"],
                       int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"]),
                       int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"])))
    ks.sort()

    # cut into segments at union-idle gaps
    segs = []
    cur = [ks[0]]
    end = ks[0][1]
    for k in ks[1:]:
        if k[0] - end > args.gap_us * 1000:
            segs.append(cur)
            cur = []
        cur.append(k)
        end = max(end, k[1])
    segs.append(cur)

    # decode tokens = the trailing segments with the full kernel count (drops the warmup fragments)
    n_full = max(len(s) for s in segs[-args.tokens:])
    tail = [s for s in segs if len(s) >= 0.9 * n_full][-args.tokens:]
    print(f"segments: {len(segs)}; kernel counts of the last {len(tail)}: min {min(len(s) for s in tail)} "
          f"max {max(len(s) for s in tail)}")

    per_tok_busy, per_tok_span, per_tok_n = [], [], []
    by_name = collections.defaultdict(list)
    by_agent_busy = collections.Counter()
    for s in tail:
        t0 = s[0][0]
        t1 = max(k[1] for k in s)
        # union busy across agents
        busy = 0
        b_end = -1
        for k in sorted(s):
            st, en = k[0], k[1]
            if st > b_end:
                busy += en - st
                b_end = en
            elif en > b_end:
                busy += en - b_end
                b_end = en
        per_tok_busy.append(busy / 1e3)
        per_tok_span.append((t1 - t0) / 1e3)
        per_tok_n.append(len(s))
        for k in s:
            nm = k[3] if args.full_names else short(k[3])
            by_name[nm].append((k[1] - k[0]) / 1e3)
            by_agent_busy[k[2]] += (k[1] - k[0]) / 1e3

    if args.dump:
        s = tail[-1]
        t0 = s[0][0]
        for k in s[:args.dump]:
            print(f"{(k[0] - t0) / 1e3:9.1f} {(k[1] - k[0]) / 1e3:6.1f} {k[2][-1]} grid {k[4] // max(k[5], 1):6d} "
                  f"{short(k[3])[:80]}")
        return

    n = len(tail)
    span = statistics.median(per_tok_span)
    busy = statistics.median(per_tok_busy)
    ksum = sum(sum(v) for v in by_name.values()) / n
    print(f"per token (median over {n}): span {span:.0f} us, union busy {busy:.0f} us "
          f"({100 * busy / span:.1f}%), idle {span - busy:.0f} us, kernels {statistics.median(per_tok_n):.0f}, "
          f"kernel-time sum {ksum:.0f} us")
    for a, v in sorted(by_agent_busy.items()):
        print(f"  {a}: kernel time {v / n:.0f} us/token")
    # gaps between consecutive kernels on the same GPU (dispatch latency), excluding hand-offs to the other GPU
    gaps = []
    for s in tail:
        last = {}
        for k in s:
            if k[2] in last:
                g = (k[0] - last[k[2]]) / 1e3
                if g < 50:
                    gaps.append(g)
            last[k[2]] = k[1]
    gaps.sort()
    print(f"  same-GPU inter-kernel gaps < 50 us: {len(gaps) / n:.0f}/token, sum {sum(gaps) / n:.0f} us/token, "
          f"p50 {gaps[len(gaps) // 2]:.2f} us, p90 {gaps[int(len(gaps) * 0.9)]:.2f} us")
    print(f"\n{'us/tok':>8} {'calls':>6} {'p50 us':>7}  kernel")
    rows = sorted(by_name.items(), key=lambda kv: -sum(kv[1]))
    for nm, v in rows[:args.top]:
        print(f"{sum(v) / n:8.1f} {len(v) / n:6.1f} {statistics.median(v):7.1f}  {nm}")
    # short-kernel floor: how much time is spent in kernels under 5 us
    small = [d for v in by_name.values() for d in v if d < 5.0]
    print(f"\nkernels < 5 us: {len(small) / n:.0f}/token, {sum(small) / n:.0f} us/token")


if __name__ == "__main__":
    main()
