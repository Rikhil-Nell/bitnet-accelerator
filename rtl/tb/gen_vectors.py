#!/usr/bin/env python3
"""Generate simulation vectors for tb_gemv_top.sv.

Writes into the output directory:
  mem.hex    initial memory image ($readmemh, 64-bit words, @word-address lines)
  exp.hex    expected memory image for the y regions
  cases.txt  one line per command: mode nb x_beats row_beats N x_addr w0..w3 y0..y3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bitnet"))
from engine_format import (NU, align, gemv, pack_int8, pack_t5, t5_k,  # noqa: E402
                           unit_slices)

BASE = 0x0010_0000
MEM_BYTES = 4 << 20   # must match MEM_WORDS in the testbench

# (mode, K, N, nb): mode 0 = T5 ternary, 1 = int8; nb = activation vectors (2: T5 only).
CASES = [
    (0, 80, 8, 1),        # 2 T5 beats per row, 2 rows per unit
    (0, 40, 64, 1),       # 1 beat per row: back-to-back row completions
    (0, 64, 8, 1),        # K not a multiple of 40: zero padding to 80
    (0, 2560, 40, 1),     # model K (attn / gate / up)
    (0, 6912, 16, 1),     # model K (ffn_down): padded to 6920, deepest activation use
    (1, 256, 24, 1),      # int8
    (1, 2560, 8, 1),      # int8 at model K (exact-head mode)
    (0, 2560, 8, 2),      # two vectors (speculative verification)
    (0, 40, 64, 2),       # two vectors, 1 beat per row: one output word per clock-ish
    (0, 6912, 16, 2),     # two vectors at the deepest K
    (0, 2560, 8, 1),      # back to one vector
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    mem: dict[int, int] = {}
    exp: dict[int, int] = {}
    lines = []
    ptr = BASE

    def put(words: np.ndarray, addr: int, into: dict[int, int]) -> None:
        for i, w in enumerate(words.tolist()):
            into[(addr >> 3) + i] = int(w)

    for mode, k, n, nb in CASES:
        assert n % (2 * NU) == 0
        x = rng.integers(-128, 128, size=k, dtype=np.int64).astype(np.int8)
        x2 = rng.integers(-128, 128, size=k, dtype=np.int64).astype(np.int8)
        x[0], x[-1] = -128, 127   # extremes
        if mode == 0:
            w = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n, k), p=[0.3, 0.4, 0.3])
        else:
            w = rng.integers(-128, 128, size=(n, k), dtype=np.int64).astype(np.int8)
            w[0, :] = -128        # worst-case magnitude row
        y = gemv(w, x)
        y2 = gemv(w, x2)

        kp = t5_k(k) if mode == 0 else k
        xb = kp // 8
        rb = kp // 40 if mode == 0 else k // 8
        xpad = np.zeros(kp * nb, dtype=np.int8)
        xpad[:k] = x
        if nb == 2:
            xpad[kp : kp + k] = x2
        x_addr = ptr = align(ptr, 128)
        put(np.frombuffer(xpad.tobytes(), dtype="<u8"), x_addr, mem)
        ptr += kp * nb
        w_addrs, y_addrs = [], []
        for sl in unit_slices(n):
            ptr = align(ptr)
            w_addrs.append(ptr)
            words = (np.frombuffer(pack_t5(w[sl]).tobytes(), dtype="<u8") if mode == 0
                     else pack_int8(w[sl]))
            put(words, ptr, mem)
            ptr += words.size * 8
        for sl in unit_slices(n):
            ptr = align(ptr)
            y_addrs.append(ptr)
            ys = y[sl] if nb == 1 else np.stack([y[sl], y2[sl]], axis=1).reshape(-1)   # rows {a, b}
            yw = np.frombuffer(ys.astype("<i4").tobytes(), dtype="<u8")
            put(yw, ptr, exp)
            # poison the output area so stale data can't pass
            put(np.full(yw.size, 0xDEADBEEFDEADBEEF, dtype=np.uint64), ptr, mem)
            ptr += yw.size * 8
        lines.append(" ".join(str(v) for v in (mode, nb, xb, rb, n, x_addr, *w_addrs, *y_addrs)))
        assert len(w_addrs) == NU

    if ptr > MEM_BYTES:
        raise SystemExit(f"vectors need {ptr:#x} bytes > MEM_BYTES")

    def dump(d: dict[int, int], path: Path) -> None:
        out, prev = [], None
        for a in sorted(d):
            if prev is None or a != prev + 1:
                out.append(f"@{a:x}")
            out.append(f"{d[a]:016x}")
            prev = a
        path.write_text("\n".join(out) + "\n")

    dump(mem, args.out / "mem.hex")
    dump(exp, args.out / "exp.hex")
    (args.out / "cases.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(CASES)} cases, {ptr - BASE:,} bytes of memory -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
