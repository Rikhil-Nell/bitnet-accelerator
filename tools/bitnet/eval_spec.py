#!/usr/bin/env python3
"""How many tokens per weight pass could speculative decoding give?

Replays saved greedy generations (eval_coarse_head.py --collect) and simulates
drafting from the context itself ("prompt lookup"): find the longest recent n-gram
suffix that occurred earlier in prompt + output, propose what followed it. With a
batch of B positions per pass (1 real + B-1 drafts), a pass yields 1 + the number of
leading drafts that match the model's actual greedy tokens. Greedy output is
unchanged by construction.

  python tools/bitnet/eval_spec.py [--ngram-table path.npz]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

STATES = Path("D:/xilinx-tools/bitnet_head_states.npz")


def load_runs(path: Path) -> list[tuple[list[int], list[int]]]:
    d = np.load(path)
    p_ids, o_ids = d["prompt_ids"].tolist(), d["out_ids"].tolist()
    runs, pi, oi = [], 0, 0
    for pl, ol in zip(d["prompt_lens"].tolist(), d["out_lens"].tolist()):
        runs.append((p_ids[pi : pi + pl], o_ids[oi : oi + ol]))
        pi += pl
        oi += ol
    return runs


def lookup_draft(ctx: list[int], k: int, max_n: int = 4) -> list[int]:
    """Longest suffix (n = max_n..1) of ctx seen earlier; return up to k following tokens."""
    for n in range(min(max_n, len(ctx) - 1), 0, -1):
        suf = ctx[-n:]
        for s in range(len(ctx) - n - 1, -1, -1):   # most recent occurrence first
            if ctx[s : s + n] == suf:
                return ctx[s + n : s + n + k]
    return []


class NGramTable:
    """Static n-gram continuation table (n-1 context tokens -> most frequent next)."""

    def __init__(self, seqs: list[list[int]], n: int = 3):
        self.n = n
        counts: dict[int, dict[tuple, dict[int, int]]] = {m: defaultdict(lambda: defaultdict(int)) for m in range(1, n)}
        for s in seqs:
            for i in range(1, len(s)):
                for m in range(1, n):
                    if i - m >= 0:
                        counts[m][tuple(s[i - m : i])][s[i]] += 1
        self.best = {m: {c: max(nx.items(), key=lambda kv: kv[1])[0] for c, nx in counts[m].items()} for m in counts}

    def draft(self, ctx: list[int], k: int) -> list[int]:
        out, c = [], list(ctx)
        for _ in range(k):
            nxt = None
            for m in range(self.n - 1, 0, -1):
                nxt = self.best[m].get(tuple(c[-m:]))
                if nxt is not None:
                    break
            if nxt is None:
                break
            out.append(nxt)
            c.append(nxt)
        return out


def simulate(runs, B: int, drafter) -> tuple[int, int]:
    """Decode-phase replay. out[0] comes from prefill. A pass feeds [out[i], d1..d(B-1)]
    where the drafts guess out[i+1..]. Position 0 always yields out[i+1]; position j
    yields out[i+1+j] if d1..dj all matched. Returns (tokens decided, passes)."""
    toks = passes = 0
    for prompt, out in runs:
        i = 0
        while i < len(out) - 1:
            draft = drafter(prompt + out[: i + 1], B - 1) if B > 1 else []
            m = 0
            while m < len(draft) and i + 1 + m < len(out) - 1 and draft[m] == out[i + 1 + m]:
                m += 1
            i += 1 + m
            toks += 1 + m
            passes += 1
    return toks, passes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=STATES)
    ap.add_argument("--corpus", type=Path, help="npz with a flat 'ids' array and 'lens' for an n-gram table")
    args = ap.parse_args()
    runs = load_runs(args.states)
    print(f"{len(runs)} runs, {sum(len(o) for _, o in runs)} generated tokens")

    def lookup(ctx, k):
        return lookup_draft(ctx, k)

    drafters = {"prompt lookup": lookup}
    if args.corpus:
        d = np.load(args.corpus)
        ids, lens = d["ids"].tolist(), d["lens"].tolist()
        seqs, o = [], 0
        for L in lens:
            seqs.append(ids[o : o + L])
            o += L
        table = NGramTable(seqs, n=4)

        def ngram(ctx, k):
            return table.draft(ctx, k)

        def combo(ctx, k):
            return lookup(ctx, k) or ngram(ctx, k)

        drafters["n-gram table"] = ngram
        drafters["lookup, else n-gram"] = combo

    for name, fn in drafters.items():
        row = []
        for B in (1, 2, 3, 4):
            t, p = simulate(runs, B, fn)
            row.append(f"B={B}: {t / p:.2f} tok/pass")
        print(f"{name:22s} " + "   ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
