#!/usr/bin/env python3
"""Smaller coarse head: rank the vocabulary in a projected space.

coarse(h) = T5(W V_r) . int8(V_r^T h), V_r = top-r right singular vectors of the head.
Bytes per pass: r/2560 of the full T5 coarse head (+ the small int8 projection).
Measures how often the exact int8 argmax stays in the coarse top-K.

  python tools/bitnet/eval_proj_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board_image import parse  # noqa: E402
from pack_board import DEFAULT_OUT  # noqa: E402

from paths import WORK  # noqa: E402

STATES = WORK / "bitnet_head_states.npz"


def ternary_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = w.abs().mean(1, keepdim=True)
    t = torch.where(w > 0.5 * m, 1.0, torch.where(w < -0.5 * m, -1.0, 0.0))
    sc = (w.abs() * t.abs()).sum(1) / t.abs().sum(1).clamp(min=1)
    return t, sc


def q8(x: torch.Tensor) -> torch.Tensor:
    s = 127.0 / x.abs().amax(-1, keepdim=True).clamp(min=1e-9)
    return torch.round(x * s)


def main() -> int:
    dev = torch.device("cuda")
    img = np.memmap(DEFAULT_OUT, dtype=np.uint8, mode="r")
    h, _ = parse(img)
    V, E = h.vocab, h.n_embd
    hq = torch.from_numpy(np.asarray(img[h.head_off : h.head_off + V * E])).to(dev).view(torch.int8).view(V, E).float()
    hs = torch.from_numpy(np.frombuffer(img[h.head_scale_off : h.head_scale_off + 4 * V].tobytes(), dtype="<f4").copy()).to(dev)
    W = hq * hs[:, None]                                    # real head
    x = torch.from_numpy(np.load(STATES)["hq"].astype(np.float32)).to(dev)   # (N, E) int8 states
    exact = (x @ hq.t()) * hs[None, :]
    best = exact.argmax(1)
    N = x.shape[0]
    # baseline: full-width T5 coarse head
    t, sc = ternary_rows(hq)
    base = (x @ t.t()) * (sc * hs)[None, :]
    rank = (base > base[torch.arange(N), best][:, None]).sum(1)
    print(f"full T5 coarse (2560): top16 {(rank < 16).float().mean() * 100:.1f}%  top64 {(rank < 64).float().mean() * 100:.1f}%  worst {int(rank.max())}")
    # SVD of the real head (on a row sample is enough for the subspace)
    _, _, Vt = torch.linalg.svd(W[::4], full_matrices=False)
    for r in (640, 1024, 1280, 1600, 2000):
        Vr = Vt[:r].t()                                     # (E, r)
        P = q8(Vr.t()) / (127.0 / Vr.t().abs().amax(1, keepdim=True))   # int8 projection rows, dequantized
        z = q8(x @ P.t())                                   # int8 projected state (scale irrelevant for ranking)
        Wr = W @ Vr                                         # (V, r) head in the subspace
        tr, scr = ternary_rows(Wr)
        coarse = (z @ tr.t()) * scr[None, :]
        rank = (coarse > coarse[torch.arange(N), best][:, None]).sum(1)
        mb = V * r / 5 / 1e6
        print(f"r={r:4d} ({mb:5.1f} MB): top16 {(rank < 16).float().mean() * 100:5.1f}%  top64 {(rank < 64).float().mean() * 100:5.1f}%  "
              f"top256 {(rank < 256).float().mean() * 100:5.1f}%  worst {int(rank.max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
