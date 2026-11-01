#!/usr/bin/env python3
"""Data formats of the PL GEMV engine (rtl/gemv_unit.v). Single source of truth
for the weight packer, the simulation vectors and the board tests.

Ternary: row-major, 32 trits per little-endian 64-bit word, trit j of a row in
bits [2j+1:2j] of word j//32, signed 2-bit code 00=0, 01=+1, 11=-1.
Int8:    row-major, 8 weights per 64-bit word (plain little-endian bytes).
A matrix with N rows is split into NU equal row slices, one per engine unit.
"""

from __future__ import annotations

import numpy as np

NU = 4             # engine units (one per HP port)
ALIGN = 4096       # slice alignment in DDR
CODE = {0: 0, 1: 1, -1: 3}


def check_shape(n: int, k: int) -> None:
    if k % 32 or not 32 <= k <= 8192:
        raise ValueError(f"K={k}: must be a multiple of 32 in [32, 8192]")
    if n % (2 * NU):
        raise ValueError(f"N={n}: must split into {NU} even row slices")


def pack_ternary(trits: np.ndarray) -> np.ndarray:
    """(N, K) int8 in {-1,0,1} -> uint64 words, row-major."""
    n, k = trits.shape
    if k % 32:
        raise ValueError(f"K={k} not a multiple of 32")
    codes = (trits.astype(np.int64) & 0x3).astype(np.uint64)  # -1 -> 3, 1 -> 1, 0 -> 0
    codes = codes.reshape(n, k // 32, 32)
    shifts = (2 * np.arange(32, dtype=np.uint64))
    return np.bitwise_or.reduce(codes << shifts, axis=-1).reshape(-1)


def unpack_ternary(words: np.ndarray, n: int, k: int) -> np.ndarray:
    w = words.reshape(n, k // 32, 1)
    codes = (w >> (2 * np.arange(32, dtype=np.uint64))) & np.uint64(3)
    t = codes.astype(np.int8).reshape(n, k)
    t[t == 3] = -1
    return t


def pack_int8(w: np.ndarray) -> np.ndarray:
    """(N, K) int8 -> uint64 words, row-major."""
    if w.shape[1] % 8:
        raise ValueError("K not a multiple of 8")
    return np.frombuffer(np.ascontiguousarray(w, dtype=np.int8).tobytes(), dtype="<u8").copy()


def pack_x(x: np.ndarray) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(x, dtype=np.int8).tobytes(), dtype="<u8").copy()


def unit_slices(n: int) -> list[slice]:
    r = n // NU
    return [slice(u * r, (u + 1) * r) for u in range(NU)]


def gemv(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Exact integer reference: y[r] = sum_k w[r,k] * x[k]."""
    return (w.astype(np.int64) @ x.astype(np.int64)).astype(np.int32)


def align(addr: int, a: int = ALIGN) -> int:
    return (addr + a - 1) // a * a


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(8, 64)).astype(np.int8)
    assert np.array_equal(unpack_ternary(pack_ternary(t), 8, 64), t)
    print("engine_format self-check OK")
