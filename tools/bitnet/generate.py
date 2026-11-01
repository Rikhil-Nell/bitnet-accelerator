#!/usr/bin/env python3
"""Generate a reply from BitNet b1.58 2B-4T with BitLinear on a chosen accelerator backend.

  python tools/bitnet/generate.py -p "What is the capital of France?" -n 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import BitNetEngine, EmulatedAccelerator  # noqa: E402
from gguf_model import DEFAULT_GGUF, load_model  # noqa: E402

HF_DIR = Path(os.environ.get("BITNET_HF_DIR", "C:/AI/models/BitNet-b1.58-2B-4T"))
STOP_IDS = {128001, 128009}


def chat_prompt(user: str) -> str:
    return f"<|begin_of_text|>User: {user.strip()}<|eot_id|>Assistant: "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-n", "--max-tokens", type=int, default=20)
    ap.add_argument("--backend", choices=["emulated"], default="emulated")
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--layers", type=int, default=None, help="debug: run only the first N layers")
    ap.add_argument("--log", type=Path, default=None, help="write a JSON record of the run")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    ids = tok.encode(chat_prompt(args.prompt), add_special_tokens=False).ids

    t0 = time.perf_counter()
    model = load_model(args.gguf, n_layer_limit=args.layers)
    backend = EmulatedAccelerator(model)
    eng = BitNetEngine(model, backend)
    print(f"[load] {time.perf_counter() - t0:.1f}s  backend={backend.name}  prompt_tokens={len(ids)}", flush=True)

    out_ids: list[int] = []
    step_times = []
    t_start = time.perf_counter()
    logits = eng.forward(ids)
    step_times.append(time.perf_counter() - t_start)
    print(f"[prefill] {step_times[-1]:.1f}s", flush=True)
    for _ in range(args.max_tokens):
        nxt = int(np.argmax(logits))
        if nxt in STOP_IDS:
            break
        out_ids.append(nxt)
        print(tok.decode([nxt]), end="", flush=True)
        t = time.perf_counter()
        logits = eng.forward([nxt])
        step_times.append(time.perf_counter() - t)
    total = time.perf_counter() - t_start
    text = tok.decode(out_ids)
    print()
    print(f"[done] {len(out_ids)} tokens in {total:.1f}s; bitlinear {eng.bitlinear_time:.1f}s; "
          f"MACs={backend.macs:,}", flush=True)

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps({
            "prompt": args.prompt,
            "backend": backend.name,
            "prompt_ids": ids,
            "output_ids": out_ids,
            "output_text": text,
            "step_seconds": step_times,
            "bitlinear_seconds": eng.bitlinear_time,
            "macs": backend.macs,
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
