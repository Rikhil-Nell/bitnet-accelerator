#!/usr/bin/env python3
"""Lightweight eval runner for golden + future sim log compare."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tools" / "golden" / "ternary_gemv.py"


def run_golden() -> int:
    print("== G0: golden system vector ==")
    return subprocess.call([sys.executable, str(GOLDEN)])


def run_pack_roundtrip() -> bool:
    print("== G1: pack/decode roundtrip ==")
    sys.path.insert(0, str(GOLDEN.parent))
    from ternary_gemv import decode_weight_word, pack_weight_word

    trits = [1, 0, -1, 1, 0, 1, -1, 0]
    word = pack_weight_word(trits)
    back = decode_weight_word(word)
    ok = back == trits
    print(f"trits={trits} word=0x{word:04x} back={back} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    rc = run_golden()
    if rc != 0:
        return rc
    if not run_pack_roundtrip():
        return 1
    print("All L0 evals passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
