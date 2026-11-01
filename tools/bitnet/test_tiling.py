#!/usr/bin/env python3
"""Prove 8x8 tile GEMM (firmware layout) matches NumPy for BitNet matrix shapes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_weights import pack_tiles  # noqa: E402


def tiled_gemm(w_words: np.ndarray, n: int, k: int, x: np.ndarray) -> np.ndarray:
    """Same math as fpga/ps/bitlinear_server/main.c gemm()."""
    ng, nb = n // 8, k // 8
    y = np.zeros(n, dtype=np.int32)
    for cb in range(nb):
        for rg in range(ng):
            tile = w_words[(cb * ng + rg) * 8 : (cb * ng + rg + 1) * 8]
            for r in range(8):
                word = int(tile[r])
                for j in range(8):
                    code = (word >> (2 * j)) & 3
                    tri = (0, 1, -1)[code] if code < 3 else 0
                    y[rg * 8 + r] += int(x[cb * 8 + j]) * tri
    return y


def main() -> int:
    rng = np.random.default_rng(0)
    shapes = [(640, 2560), (2560, 2560), (6912, 2560), (2560, 6912)]
    for n, k in shapes:
        trits = rng.integers(-1, 2, size=(n, k), dtype=np.int8)
        words = pack_tiles(trits)
        x = rng.integers(-128, 128, size=k, dtype=np.int8)
        got = tiled_gemm(words, n, k, x)
        expect = (trits.astype(np.int64) @ x.astype(np.int64)).astype(np.int32)
        if not np.array_equal(got, expect):
            print(f"FAIL {n}x{k}")
            return 1
        print(f"PASS tiled GEMM {n}x{k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
