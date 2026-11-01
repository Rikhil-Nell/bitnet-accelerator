#!/usr/bin/env python3
"""
Capstone BitLinear kernel (W2A8 ternary weights, INT8 activations).
Behavior matches bitnet_accelerator RTL + tools/golden/ternary_gemv.py.
Use this as the drop-in linear op on PC until hardware MMIO is wired.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "golden"))

from ternary_gemv import decode_weight_word, gemv, pack_weight_word, TRIT_FROM_CODE


def trits_to_int8_matrix(rows: list[list[int]]) -> list[list[int]]:
    """Expand ternary rows to signed int8 weight matrix for reference matmul."""
    return [[int(w) for w in row] for row in rows]


def weight_words_to_matrix(weight_words: list[int]) -> list[list[int]]:
    return [decode_weight_word(w) for w in weight_words]


def bitlinear_gemv_int8(
    activations: list[int],
    weight_words: list[int],
) -> list[int]:
    """Same as RTL fixed 8x8 engine (one output vector)."""
    return gemv(activations, weight_words)


def bitlinear_gemv_numpy(activations, weight_words):
    """Reference: int8 activations @ ternary W^T using NumPy if available."""
    import numpy as np

    x = np.array(activations, dtype=np.int32)
    w = np.array(weight_words_to_matrix(weight_words), dtype=np.int32)
    y = w @ x
    return y.astype(int).tolist()


def i2s_word_to_capstone_word(i2s_word: int) -> int:
    """Convert one 16-bit capstone-packed word from I2_S-style 2-bit codes per nibble lane."""
    out = 0
    for i in range(8):
        code = (i2s_word >> (2 * i)) & 0x3
        if code == 0:
            tri = -1
        elif code == 1:
            tri = 0
        elif code == 2:
            tri = 1
        else:
            tri = 0
        if tri == 0:
            c = 0
        elif tri == 1:
            c = 1
        else:
            c = 2
        out |= c << (2 * i)
    return out & 0xFFFF
