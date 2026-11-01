#!/usr/bin/env python3
"""Per-layer residual-stream trace from the packed image, in the same format the
firmware prints after a 'V' command ("# L <layer> pos <p> x0=.. x1=.. mean|x|=..").

  python tools/bitnet/layer_trace.py --ids 128000 1502 25 ...

Compare line by line with the board's output to find the first diverging layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
from board_emu import load_image  # noqa: E402
from pack_board import DEFAULT_OUT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--layers", type=int, default=None)
    args = ap.parse_args()

    model, head_q, head_scale = load_image(args.image, n_layer_limit=args.layers)
    eng = E.BitNetEngine(model, E.EmulatedAccelerator(model))
    eng.head_int8, eng.head_q, eng.head_scale = True, head_q, head_scale

    # forward() calls rms_norm 4x per layer (attn, attn_sub, ffn, ffn_sub) and once for
    # the output norm, so the residual after layer l is the input of call 4*(l+1).
    captured: list[np.ndarray] = []
    real_norm = E.rms_norm

    def spy(x, w, eps):
        captured.append(x)
        return real_norm(x, w, eps)

    E.rms_norm = spy
    for pos, t in enumerate(args.ids):
        captured.clear()
        logits = eng.forward([t])
        for l in range(model.n_layer):
            x = captured[4 * (l + 1)][-1]
            print(f"# L {l} pos {pos} x0={x[0]:.4f} x1={x[1]:.4f} mean|x|={np.mean(np.abs(x)):.4f}")
    print(f"next token: {int(np.argmax(logits))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
