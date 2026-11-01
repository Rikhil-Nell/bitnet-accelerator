#!/usr/bin/env python3
"""Run make sim-all via Icarus if present, else Vivado xsim batch."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def repo_for_vivado() -> Path:
    """Vivado on Windows mishandles spaces in RTL paths; use SUBST drive if needed."""
    if " " not in str(ROOT):
        return ROOT
    drive = "V:"
    home = str(Path.home())
    if not Path(f"{drive}/").exists():
        subprocess.call(["subst", drive, home], shell=True)
    rel = ROOT.relative_to(Path.home())
    short = Path(f"{drive}/{rel.as_posix()}")
    if short.is_dir():
        return short
    return ROOT


def run_iverilog() -> int:
    make = shutil.which("make") or "make"
    for target in ("sim-decoder", "sim-mac", "sim-accelerator"):
        rc = subprocess.call([make, target], cwd=ROOT)
        if rc != 0:
            return rc
    print("Unit sim smoke tests finished (Icarus).")
    return 0


def run_xsim() -> int:
    vivado_root = repo_for_vivado()
    env_ps1 = ROOT / "tools/setup/env.ps1"
    (ROOT / "build/sim").mkdir(parents=True, exist_ok=True)
    log = ROOT / "build/sim/xsim.log"
    jou = ROOT / "build/sim/xsim.jou"
    cmd = (
        f". '{env_ps1}'; $env:BITNET_REPO_ROOT='{vivado_root}'; "
        f"Set-Location '{vivado_root}'; "
        f"vivado -mode batch -source fpga/scripts/sim_rtl_xsim.tcl "
        f"-log '{log}' -journal '{jou}'"
    )
    return subprocess.call(["powershell", "-NoProfile", "-Command", cmd])


def main() -> int:
    if shutil.which("iverilog"):
        return run_iverilog()
    print("iverilog not in PATH — falling back to Vivado xsim (see build/sim/xsim*.log)")
    return run_xsim()


if __name__ == "__main__":
    raise SystemExit(main())
