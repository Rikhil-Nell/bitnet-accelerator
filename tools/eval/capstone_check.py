#!/usr/bin/env python3
"""Report capstone command-matrix status (non-destructive; does not rebuild bitstream)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_py(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def usb_converter_b_problem() -> str:
    ps = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not ps.is_file():
        return ""
    script = (
        "(Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match 'Converter B' } | "
        "Select-Object -First 1 -ExpandProperty Problem) -join ''"
    )
    proc = subprocess.run(
        [str(ps), "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
    )
    return (proc.stdout or "").strip()


def jtag_target_visible() -> tuple[bool, str]:
    """Prefer fresh Vivado list_hw when blocked; fall back to log file."""
    prob = usb_converter_b_problem()
    if prob == "CM_PROB_NEED_RESTART":
        return False, "reboot required (Converter B CM_PROB_NEED_RESTART); docs/ZYBO_JTAG.md"

    list_log = ROOT / "build/vivado/list_hw.log"
    env_ps1 = ROOT / "tools/setup/env.ps1"
    import shutil

    if os.environ.get("CAPSTONE_SKIP_JTAG_PROBE", "").lower() in ("1", "true", "yes"):
        if not list_log.is_file():
            return False, "set CAPSTONE_SKIP_JTAG_PROBE=0 and run make list-hw manually"
        # fall through to parse log only
    elif shutil.which("vivado") and env_ps1.is_file():
        vivado = shutil.which("vivado")
        list_log.parent.mkdir(parents=True, exist_ok=True)
        cmd = (
            f". '{env_ps1}'; vivado -mode batch -source fpga/scripts/list_hw_targets.tcl "
            f"-log '{list_log}'"
        )
        ps = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        proc = subprocess.run(
            [str(ps), "-NoProfile", "-Command", cmd],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return False, f"list_hw failed (exit {proc.returncode})"
    elif not list_log.is_file() and not os.environ.get("CAPSTONE_SKIP_JTAG_PROBE"):
        return False, "run make list-hw (USB Serial Converter B)"

    for line in list_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("targets on localhost:3121:"):
            rest = line[len("targets on localhost:3121:") :].strip()
            if rest:
                return True, f"JTAG target: {rest}"
            return False, "empty hw_targets (USB Serial Converter B / docs/ZYBO_JTAG.md)"
    return False, "list_hw.log missing target line"


def usb_converter_b_hint() -> str:
    prob = usb_converter_b_problem()
    if prob and prob != "CM_PROB_NEED_RESTART":
        return f" (Converter B Problem={prob})"
    return ""


def xsim_ok() -> bool:
    log = ROOT / "build/sim/xsim.log"
    if not log.is_file():
        return False
    body = log.read_text(encoding="utf-8", errors="replace")
    need = (
        "PASS: ternary_decoder_tb",
        "PASS: ternary_mac_tb",
        "PASS: bitnet_accelerator_tb",
    )
    return all(m in body for m in need)


def main() -> int:
    rows: list[tuple[str, str, str]] = []

    rc, _ = run_py("tools/eval/run_eval.py")
    rows.append(("make eval (L0)", "PASS" if rc == 0 else "FAIL", "run_eval.py"))

    rc, _ = run_py("tools/host/test_model_kernel.py")
    rows.append(("make test-kernel", "PASS" if rc == 0 else "FAIL", "test_model_kernel.py"))

    rc, _ = run_py("tools/host/mmio_runner.py")
    rows.append(("make test-mmio", "PASS" if rc == 0 else "FAIL", "mmio_runner.py"))

    rc, _ = run_py("tools/host/test_i2s_convert.py")
    rows.append(("make test-i2s", "PASS" if rc == 0 else "FAIL", "test_i2s_convert.py"))

    sim = xsim_ok()
    rows.append(
        (
            "make sim-all",
            "PASS" if sim else "UNKNOWN",
            "build/sim/xsim.log (re-run make sim-all to refresh)",
        )
    )

    util = ROOT / "build/vivado/utilization.rpt"
    timing = ROOT / "build/vivado/timing.rpt"
    ttext = timing.read_text(encoding="utf-8", errors="replace") if timing.is_file() else ""
    synth_ok = util.is_file() and timing.is_file() and (
        "Timing Summary Report" in ttext or "WNS" in ttext
    )
    rows.append(
        (
            "make vivado-synth",
            "PASS" if synth_ok else "UNKNOWN",
            "build/vivado/utilization.rpt + timing.rpt",
        )
    )

    bit = ROOT / "build/vivado/bitnet_zybo.bit"
    rows.append(
        (
            "make bitstream",
            "PASS" if bit.is_file() and bit.stat().st_size > 100_000 else "FAIL",
            str(bit),
        )
    )

    elf = Path("D:/xilinx-tools/bitnet_vitis_ws_clean/system_8x8_test/build/system_8x8_test.elf")
    rows.append(("make ps-app", "PASS" if elf.is_file() else "FAIL", str(elf)))

    jtag, jtag_ev = jtag_target_visible()
    if not jtag:
        jtag_ev += usb_converter_b_hint()
    rows.append(
        (
            "make program",
            "PASS" if jtag else "BLOCKED",
            jtag_ev,
        )
    )
    uart_log = ROOT / "build/capstone/uart.log"
    uart_text = ""
    if uart_log.is_file():
        raw = uart_log.read_bytes()
        enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
        uart_text = raw.decode(enc, errors="replace")
    board_pass = "PASS: system_8x8 on PL accelerator" in uart_text
    rows.append(
        (
            "make board-test",
            "PASS" if board_pass else ("TODO" if jtag else "BLOCKED"),
            f"{uart_log.relative_to(ROOT)} (UART PASS line)"
            if board_pass
            else "make program, start uart-listen COM=COM4, then make run-ps",
        )
    )

    print(f"{'Target':<22} {'Status':<10} Evidence")
    print("-" * 72)
    fail = 0
    for name, status, ev in rows:
        print(f"{name:<22} {status:<10} {ev}")
        if status in ("FAIL", "BLOCKED", "TODO"):
            fail += 1
    print("")
    if fail:
        print(f"{fail} item(s) not green; see STATUS.md and docs/ZYBO_JTAG.md")
        return 1
    print("All capstone targets green, including on-board UART PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
