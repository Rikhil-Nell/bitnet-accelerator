#!/usr/bin/env python3
"""Generate simulation vectors for tb_gemv_top.sv.

Writes into the output directory:
  mem.hex    initial memory image ($readmemh, 64-bit words, @word-address lines)
  exp.hex    expected memory image for the y regions
  cases.txt  one line per command: mode K N x_addr w0..w3 y0..y3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bitnet"))
from engine_format import (NU, align, check_shape, gemv, pack_int8, pack_ternary,  # noqa: E402
                           pack_x, unit_slices)

BASE = 0x0010_0000
MEM_BYTES = 4 << 20   # must match MEM_WORDS in the testbench

# (mode, K, N): small corner cases plus the model's real K values
CASES = [
    (0, 64, 8),        # smallest: 2 rows per unit, 2 beats per row
    (0, 32, 64),       # 1 beat per row: back-to-back row completions
    (0, 2560, 40),     # model K (attn / gate / up)
    (0, 6912, 16),     # model K (ffn_down), deepest activation buffer use
    (1, 256, 24),      # int8 (LM head mode)
    (1, 2560, 8),      # int8 at model K
    (0, 2560, 8),      # back-to-back command reusing the engine
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

    for mode, k, n in CASES:
        check_shape(n, k)
        x = rng.integers(-128, 128, size=k, dtype=np.int64).astype(np.int8)
        x[0], x[-1] = -128, 127   # extremes
        if mode == 0:
            w = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n, k), p=[0.3, 0.4, 0.3])
        else:
            w = rng.integers(-128, 128, size=(n, k), dtype=np.int64).astype(np.int8)
            w[0, :] = -128        # worst-case magnitude row
        y = gemv(w, x)

        x_addr = ptr = align(ptr, 128)
        put(pack_x(x), x_addr, mem)
        ptr += k
        w_addrs, y_addrs = [], []
        for sl in unit_slices(n):
            ptr = align(ptr)
            w_addrs.append(ptr)
            words = pack_ternary(w[sl]) if mode == 0 else pack_int8(w[sl])
            put(words, ptr, mem)
            ptr += words.size * 8
        for sl in unit_slices(n):
            ptr = align(ptr)
            y_addrs.append(ptr)
            yw = np.frombuffer(y[sl].astype("<i4").tobytes(), dtype="<u8")
            put(yw, ptr, exp)
            # poison the output area so stale data can't pass
            put(np.full(yw.size, 0xDEADBEEFDEADBEEF, dtype=np.uint64), ptr, mem)
            ptr += yw.size * 8
        lines.append(" ".join(str(v) for v in (mode, k, n, x_addr, *w_addrs, *y_addrs)))
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
