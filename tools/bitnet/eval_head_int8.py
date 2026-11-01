#!/usr/bin/env python3
"""Does the int8 LM head / embedding (board format) change greedy answers?

Runs each prompt through the reference engine twice (float head vs int8 head) and
compares generated token ids.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import BitNetEngine, EmulatedAccelerator  # noqa: E402
from generate import HF_DIR, STOP_IDS, chat_prompt  # noqa: E402
from gguf_model import load_model  # noqa: E402

PROMPTS = [
    "What is the capital of France?",
    "How many legs does a spider have?",
    "Who wrote Romeo and Juliet?",
    "What is 12 times 7?",
    "Name a planet in our solar system that has rings.",
    "What gas do plants absorb from the air?",
]


def run(eng: BitNetEngine, ids: list[int], n: int) -> tuple[list[int], list[float]]:
    eng.reset()
    out, margins = [], []
    logits = eng.forward(ids)
    for _ in range(n):
        top2 = np.sort(logits)[-2:]
        margins.append(float(top2[1] - top2[0]))
        nxt = int(np.argmax(logits))
        if nxt in STOP_IDS:
            break
        out.append(nxt)
        logits = eng.forward([nxt])
    return out, margins


def main() -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    model = load_model()
    acc = EmulatedAccelerator(model)
    ref = BitNetEngine(model, acc, head_int8=False)
    q8 = BitNetEngine(model, acc, head_int8=True)
    n = int(os.environ.get("N_TOKENS", "24"))
    same = 0
    for p in PROMPTS:
        ids = tok.encode(chat_prompt(p), add_special_tokens=False).ids
        t0 = time.perf_counter()
        a, _ = run(ref, ids, n)
        b, mb = run(q8, ids, n)
        match = a == b
        same += match
        first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"\n[{'SAME' if match else 'DIFF'}] {p}  ({time.perf_counter() - t0:.0f}s)")
        print(f"  f32 : {tok.decode(a)!r}")
        print(f"  int8: {tok.decode(b)!r}")
        if not match:
            where = "stop-token choice" if first_diff == min(len(a), len(b)) else "content"
            margin = mb[first_diff] if first_diff < len(mb) else float("nan")
            print(f"  first difference at token {first_diff} ({where}); int8 top-1 margin there {margin:.3f}")
    print(f"\n{same}/{len(PROMPTS)} prompts identical over {n} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
