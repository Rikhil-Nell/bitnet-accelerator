#!/usr/bin/env python3
"""Fused gate+up tile walk matches two independent gemms (firmware gemm_fused_same)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_weights import pack_tiles  # noqa: E402
from test_tiling import tiled_gemm  # noqa: E402


def fused_same_n(words_list: list[np.ndarray], n: int, k: int, x: np.ndarray) -> np.ndarray:
    m = len(words_list)
    ng, nb = n // 8, k // 8
    y = np.zeros(m * n, dtype=np.int32)
    for cb in range(nb):
        for rg in range(ng):
            xt = x[cb * 8 : (cb + 1) * 8]
            for mi, words in enumerate(words_list):
                tile = words[(cb * ng + rg) * 8 : (cb * ng + rg + 1) * 8]
                for r in range(8):
                    word = int(tile[r])
                    for j in range(8):
                        code = (word >> (2 * j)) & 3
                        tri = (0, 1, -1)[code] if code < 3 else 0
                        y[mi * n + rg * 8 + r] += int(xt[j]) * tri
    return y


def main() -> int:
    rng = np.random.default_rng(0)
    for n, k in ((6912, 2560), (2560, 2560)):
        trits_a = rng.integers(-1, 2, size=(n, k), dtype=np.int8)
        trits_b = rng.integers(-1, 2, size=(n, k), dtype=np.int8)
        wa, wb = pack_tiles(trits_a), pack_tiles(trits_b)
        x = rng.integers(-128, 128, size=k, dtype=np.int8)
        ref = np.concatenate([tiled_gemm(wa, n, k, x), tiled_gemm(wb, n, k, x)])
        got = fused_same_n([wa, wb], n, k, x)
        if not np.array_equal(got, ref):
            print(f"FAIL fused {n}x{k}")
            return 1
        print(f"PASS fused_same_n {n}x{k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
