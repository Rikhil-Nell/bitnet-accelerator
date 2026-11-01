#!/usr/bin/env python3
"""Board bring-up check for bitlinear_server: ping, self-test, then real BitNet matrices vs NumPy.

  python tools/bitnet/board_check.py --layer 0 --proj attn_k attn_q
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from board_backend import MANIFEST, BoardLink  # noqa: E402
from gguf_model import load_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--proj", nargs="+", default=["attn_k"])
    ap.add_argument("--tokens", type=int, default=1)
    args = ap.parse_args()

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))["matrices"]
    link = BoardLink(args.port, timeout=120)
    print(f"ping v{link.ping()}  self-test {'PASS' if link.self_test() else 'FAIL'}")
    model = load_model(n_layer_limit=args.layer + 1)
    rng = np.random.default_rng(0)
    ok = True
    for p in args.proj:
        e = man[f"{args.layer}.{p}"]
        good = link.checksum(e["addr"], e["bytes"]) == e["checksum"]
        print(f"{p}: DDR checksum {'OK' if good else 'MISMATCH'}")
        if not good:
            ok = False
            continue
        w = model.layers[args.layer].proj[p].trits.astype(np.int64)
        x = rng.integers(-128, 128, size=(args.tokens, e["k"])).astype(np.int8)
        t0 = time.perf_counter()
        y, secs = link.gemm([(e["addr"], e["n"])], x)
        wall = time.perf_counter() - t0
        match = np.array_equal(y, x.astype(np.int64) @ w.T)
        tiles = e["n"] * e["k"] // 64 * args.tokens
        print(f"{p}: {e['n']}x{e['k']} T={args.tokens} match={match}  board {secs:.2f}s  wall {wall:.2f}s  "
              f"{tiles / secs / 1e3:.1f}k tile-runs/s")
        ok &= match
    print("BOARD CHECK PASS" if ok else "BOARD CHECK FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
