#!/usr/bin/env python3
"""
Run system_8x8 through the capstone BitLinear kernel (PC inference stand-in).

When hardware is available, extend with UIO/devmem at 0x43C00000 or a UART bridge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "host"))
sys.path.insert(0, str(ROOT / "tools" / "golden"))

from bitlinear_kernel import bitlinear_gemv_int8
from ternary_gemv import load_vector


def main() -> int:
    p = argparse.ArgumentParser(description="Run capstone 8x8 BitLinear vector")
    p.add_argument(
        "--vector",
        type=Path,
        default=ROOT / "tools" / "golden" / "vectors" / "system_8x8.json",
    )
    args = p.parse_args()
    vec = load_vector(args.vector)
    acts = vec["activations"]
    words = [int(h, 16) for h in vec["weight_words_hex"]]
    out = bitlinear_gemv_int8(acts, words)
    print(json.dumps({"outputs": out, "expected": vec["expected_outputs"]}, indent=2))
    ok = out == vec["expected_outputs"]
    print("PASS" if ok else "FAIL", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
