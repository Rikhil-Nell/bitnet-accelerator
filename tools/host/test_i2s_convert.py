#!/usr/bin/env python3
"""Verify I2_S -> capstone word remap (for Microsoft BitNet weight import)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "host"))
sys.path.insert(0, str(ROOT / "tools" / "golden"))

from bitlinear_kernel import i2s_word_to_capstone_word
from ternary_gemv import decode_weight_word, pack_weight_word


def main() -> int:
    # I2_S codes: 00=-1, 01=0, 10=+1 for first three trits in low 6 bits
    i2s = (0 << 0) | (1 << 2) | (2 << 4)  # -1, 0, +1 at positions 0,1,2
    cap = i2s_word_to_capstone_word(i2s)
    trits = decode_weight_word(cap)[:3]
    if trits != [-1, 0, 1]:
        print("FAIL i2s remap", trits)
        return 1
    roundtrip = pack_weight_word(trits + [0] * 5)
    if decode_weight_word(roundtrip)[:3] != trits:
        print("FAIL roundtrip")
        return 1
    print("PASS: I2_S nibble remap -> capstone encoding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
