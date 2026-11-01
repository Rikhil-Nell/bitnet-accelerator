#!/usr/bin/env python3
"""
Verify PC-side BitLinear kernel matches golden + (optional) NumPy reference + RTL sim log.
Run: python tools/host/test_model_kernel.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "host"))
sys.path.insert(0, str(ROOT / "tools" / "golden"))

from bitlinear_kernel import bitlinear_gemv_int8, bitlinear_gemv_numpy
from ternary_gemv import load_vector


def xsim_log_pass() -> bool:
    log = ROOT / "build/sim/xsim.log"
    if not log.is_file():
        return False
    body = log.read_text(encoding="utf-8", errors="replace")
    return "PASS: bitnet_accelerator_tb (xsim)" in body


def run_rtl_sim() -> bool | None:
    """Return True if sim passed, False if failed, None if iverilog unavailable."""
    import shutil

    if not shutil.which("iverilog"):
        if xsim_log_pass():
            return True
        return None
    proc = subprocess.run(
        ["make", "sim-accelerator"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    if "TEST PASSED" in out:
        return True
    if proc.returncode != 0:
        print(out[-4000:])
        return False
    return "TEST PASSED" in out


def main() -> int:
    vec_path = ROOT / "tools" / "golden" / "vectors" / "system_8x8.json"
    vec = load_vector(vec_path)
    acts = vec["activations"]
    words = [int(h, 16) for h in vec["weight_words_hex"]]
    expected = vec["expected_outputs"]

    hw_kernel = bitlinear_gemv_int8(acts, words)
    if hw_kernel != expected:
        print("FAIL: capstone kernel != golden expected")
        print(" got", hw_kernel)
        print(" exp", expected)
        return 1
    print("PASS: capstone BitLinear kernel == golden vector")

    try:
        np_out = bitlinear_gemv_numpy(acts, words)
        if np_out != expected:
            print("FAIL: numpy reference != golden")
            print(" got", np_out)
            return 1
        print("PASS: NumPy BitLinear reference == golden (model-style matmul)")
    except ImportError:
        print("SKIP: numpy not installed (optional reference)")

    import shutil

    rtl = run_rtl_sim()
    if rtl is True:
        if xsim_log_pass() and not shutil.which("iverilog"):
            print("PASS: Vivado xsim RTL (bitnet_accelerator_tb); see build/sim/xsim.log")
        else:
            print("PASS: Icarus RTL sim (bitnet_accelerator_tb)")
    elif rtl is False:
        print("FAIL: RTL sim")
        return 1
    else:
        print("SKIP: iverilog not installed; run make sim-all for xsim fallback")

    from bitnet_paths import discover_bitnet_cpp_root, discover_gguf

    root = discover_bitnet_cpp_root()
    gguf = discover_gguf()
    if root and gguf:
        print(f"NOTE: BitNet.cpp ready — run make test-bitnet-smoke")
    elif root:
        print(f"NOTE: BitNet repo at {root}; download GGUF (see docs/BITNET_PC_INFERENCE.md)")
    else:
        print("NOTE: Set BITNET_CPP_ROOT / install BitNet.cpp (docs/BITNET_PC_INFERENCE.md)")

    print("Host kernel is ready to substitute BitLinear in PC inference loops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
