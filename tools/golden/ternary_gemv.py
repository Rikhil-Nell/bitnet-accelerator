#!/usr/bin/env python3
"""Golden W2A8 ternary GEMV matching capstone RTL encoding (see docs/DESIGN.md)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

# Repo RTL encoding (NOT bitnet.cpp I2_S)
TRIT_FROM_CODE = (0, 1, -1, 0)  # codes 0..3


def decode_weight_word(word: int) -> list[int]:
    trits = []
    for i in range(8):
        code = (word >> (2 * i)) & 0x3
        trits.append(TRIT_FROM_CODE[code])
    return trits


def pack_weight_word(trits: Iterable[int]) -> int:
    word = 0
    for i, t in enumerate(trits):
        if t == 0:
            code = 0
        elif t == 1:
            code = 1
        elif t == -1:
            code = 2
        else:
            raise ValueError(f"invalid trit {t}")
        word |= code << (2 * i)
    return word & 0xFFFF


def gemv(activations: list[int], weight_words: list[int]) -> list[int]:
    if len(activations) != 8:
        raise ValueError("P0 supports K=8 activations only")
    if len(weight_words) != 8:
        raise ValueError("P0 supports N=8 output rows only")
    outputs = []
    for word in weight_words:
        row = decode_weight_word(word)
        acc = sum(a * w for a, w in zip(activations, row))
        outputs.append(acc)
    return outputs


def i2s_code_to_trit(code: int) -> int:
    """Microsoft I2_S: 00=-1, 01=0, 10=+1."""
    return (-1, 0, 1)[code] if code < 3 else 0


def load_vector(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    vec_path = Path(__file__).parent / "vectors" / "system_8x8.json"
    vec = load_vector(vec_path)
    acts = vec["activations"]
    words = [int(h, 16) for h in vec["weight_words_hex"]]
    expected = vec["expected_outputs"]
    got = gemv(acts, words)
    ok = got == expected
    print(f"vector: {vec['name']}")
    print(f"outputs: {got}")
    print(f"expected: {expected}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
