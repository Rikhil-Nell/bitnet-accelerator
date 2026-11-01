#!/usr/bin/env python3
"""Run Microsoft bitnet.cpp inference (run_inference.py) if installed."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "host"))

from bitnet_paths import discover_bitnet_cpp_root, discover_gguf, discover_model_dir


def main() -> int:
    root = discover_bitnet_cpp_root()
    if root is None:
        print("SKIP: BITNET_CPP_ROOT missing or no run_inference.py")
        print("  Clone: git clone --recursive https://github.com/microsoft/BitNet.git")
        print("  Then VS2022 dev shell: python setup_env.py -md <model_dir> -q i2_s")
        return 0

    gguf = discover_gguf()
    if gguf is None:
        model_dir = discover_model_dir()
        print("SKIP: no I2_S GGUF found.")
        print(f"  Searched env + C:/AI/models; model_dir={model_dir}")
        print("  Download: huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf")
        return 0

    script = root / "run_inference.py"
    prompt = os.environ.get("BITNET_PROMPT", "Hello")
    threads = os.environ.get("BITNET_THREADS", "8")
    cmd = [
        sys.executable,
        str(script),
        "-m",
        str(gguf),
        "-p",
        prompt,
        "-cnv",
        "-t",
        threads,
        "-n",
        "32",
    ]
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=root)
    if proc.returncode != 0:
        print(f"FAIL: run_inference exit {proc.returncode}")
        return proc.returncode
    print("PASS: bitnet.cpp inference completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
