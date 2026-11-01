#!/usr/bin/env python3
"""Can a cheaper LM head find the right next token?

The int8 head reads 328 MB per token (39 % of all bytes). Idea: score all 128K rows
with a coarse low-bit copy of the head, keep the top K candidates, and rescore only
those K rows exactly with int8. The answer is unchanged whenever the exact int8
argmax is among the coarse top K. This measures how often that holds.

  python tools/bitnet/eval_coarse_head.py --collect   # greedy runs on the emulator, saves states
  python tools/bitnet/eval_coarse_head.py             # evaluates coarse formats on the saved states
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

STATES = Path("D:/xilinx-tools/bitnet_head_states.npz")
PROMPTS = [
    "What is the capital of France?",
    "How many legs does a spider have?",
    "Who wrote Romeo and Juliet?",
    "What is 12 times 7?",
    "Name a planet in our solar system that has rings.",
    "What gas do plants absorb from the air?",
    "Explain in two sentences why the sky is blue.",
    "What is the largest ocean on Earth?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water in Celsius?",
    "Give me three tips for staying healthy.",
    "What does a CPU do in a computer?",
]


def collect(n_tokens: int) -> None:
    from board_emu import load_image
    from engine import BitNetEngine, EmulatedAccelerator, quantize_act_i8
    from generate import HF_DIR, STOP_IDS, chat_prompt
    from pack_board import DEFAULT_OUT
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    model, head_q, head_scale = load_image(DEFAULT_OUT)
    eng = BitNetEngine(model, EmulatedAccelerator(model))
    eng.head_int8, eng.head_q, eng.head_scale = True, head_q, head_scale

    states: list[np.ndarray] = []
    orig = eng.lm_head_int8

    def hook(h: np.ndarray) -> np.ndarray:
        states.append(quantize_act_i8(h)[0][0].astype(np.int8))
        return orig(h)

    eng.lm_head_int8 = hook
    seqs = []
    for p in PROMPTS:
        t0 = time.perf_counter()
        eng.reset()
        ids = tok.encode(chat_prompt(p), add_special_tokens=False).ids
        logits = eng.forward(ids)
        out = []
        for _ in range(n_tokens):
            nxt = int(np.argmax(logits))
            if nxt in STOP_IDS:
                break
            out.append(nxt)
            logits = eng.forward([nxt])
        seqs.append((ids, out))
        print(f"{time.perf_counter() - t0:5.0f}s {p!r} -> {tok.decode(out)!r}", flush=True)
    lens = np.array([len(o) for _, o in seqs])
    flat_p = np.concatenate([np.array(i) for i, _ in seqs])
    flat_o = np.concatenate([np.array(o, dtype=np.int64) for _, o in seqs]) if lens.sum() else np.zeros(0)
    np.savez(STATES, hq=np.stack(states), prompt_lens=np.array([len(i) for i, _ in seqs]),
             prompt_ids=flat_p, out_lens=lens, out_ids=flat_o)
    print(f"saved {len(states)} states to {STATES}")


def coarse_formats(head_q: np.ndarray, head_scale: np.ndarray):
    """Yield (name, bits per weight, function(rows) -> (coarse int weights, per-row float scale))."""
    def ternary(thr: float):
        def f(w):
            wf = w.astype(np.float32)
            m = np.mean(np.abs(wf), axis=1, keepdims=True) + 1e-6
            t = np.where(wf > thr * m, 1, np.where(wf < -thr * m, -1, 0)).astype(np.int8)
            # least-squares scale for the kept pattern
            num = np.sum(wf * t, axis=1)
            den = np.maximum(np.sum(np.abs(t), axis=1), 1)
            return t, num / den
        return f

    def int_bits(bits: int):
        qmax = 2 ** (bits - 1) - 1
        def f(w):
            wf = w.astype(np.float32)
            a = np.max(np.abs(wf), axis=1, keepdims=True) + 1e-6
            q = np.clip(np.round(wf / a * qmax), -qmax, qmax).astype(np.int8)
            return q, a[:, 0] / qmax
        return f

    yield "ternary absmean", 2, ternary(0.5)
    yield "ternary 0.7*mean", 2, ternary(0.7)
    yield "int3", 3, int_bits(3)
    yield "int4", 4, int_bits(4)


def evaluate(chunk: int = 8192) -> None:
    from board_emu import load_image
    from pack_board import DEFAULT_OUT

    d = np.load(STATES)
    hq = d["hq"].astype(np.float32)          # (N, E)
    N = hq.shape[0]
    _, head_q, head_scale = load_image(DEFAULT_OUT, n_layer_limit=1)
    V = head_q.shape[0]
    print(f"{N} hidden states, vocab {V}")

    exact = np.zeros((N, V), np.float32)
    for r0 in range(0, V, chunk):
        w = head_q[r0 : r0 + chunk].astype(np.float32)
        exact[:, r0 : r0 + chunk] = (hq @ w.T) * head_scale[r0 : r0 + chunk][None, :]
    best = exact.argmax(axis=1)

    Ks = [1, 4, 16, 64, 256, 1024]
    for name, bits, fn in coarse_formats(head_q, head_scale):
        coarse = np.zeros((N, V), np.float32)
        for r0 in range(0, V, chunk):
            q, sc = fn(head_q[r0 : r0 + chunk])
            coarse[:, r0 : r0 + chunk] = (hq @ q.astype(np.float32).T) * (sc * head_scale[r0 : r0 + chunk])[None, :]
        # rank of the exact winner in the coarse ordering (0 = coarse top-1)
        win = coarse[np.arange(N), best]
        rank = (coarse > win[:, None]).sum(axis=1)
        mb = V * 2560 * bits / 8 / 1e6
        rec = "  ".join(f"top{k}:{np.mean(rank < k) * 100:5.1f}%" for k in Ks)
        print(f"{name:18s} {mb:5.0f} MB  {rec}  worst rank {rank.max()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("-n", type=int, default=24)
    args = ap.parse_args()
    if args.collect:
        collect(args.n)
    evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
