#!/usr/bin/env python3
"""Run capstone kernel on HF-derived slice JSON (if present)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "host"))
sys.path.insert(0, str(ROOT / "tools" / "golden"))

from bitlinear_kernel import bitlinear_gemv_int8
from ternary_gemv import load_vector

VEC = ROOT / "tools/golden/vectors/bitnet_layer0_q_8x8.json"


def main() -> int:
    if not VEC.exists():
        print(f"SKIP: {VEC} missing — run: python tools/host/extract_bitlinear_slice.py")
        return 0
    vec = load_vector(VEC)
    acts = vec["activations"]
    words = [int(h, 16) for h in vec["weight_words_hex"]]
    expected = vec["expected_outputs"]
    got = bitlinear_gemv_int8(acts, words)
    if got != expected:
        print("FAIL: kernel vs slice expected")
        print(" got", got)
        print(" exp", expected)
        return 1
    print(f"PASS: capstone kernel matches {VEC.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
