#!/usr/bin/env python3
"""
Drive bitnet_ip_top CSRs from PC (simulation stand-in for PS MMIO @ 0x43C00000).

Uses capstone register map from docs/ITERATION.md. For hardware, replace
MemAccess with /dev/mem or a small PS driver.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "golden"))
sys.path.insert(0, str(ROOT / "tools" / "host"))

from bitlinear_kernel import bitlinear_gemv_int8
from ternary_gemv import load_vector

REG_CTRL = 0x00
REG_STATUS = 0x04
REG_ACT = 0x10
REG_WGT = 0x14
REG_OUTIDX = 0x18
REG_OUTDAT = 0x1C

BASE = 0x43C0_0000


class MockMmio:
    """Software CSR model of bitnet_axi_lite + accelerator (behavioral)."""

    def __init__(self) -> None:
        from bitlinear_kernel import bitlinear_gemv_int8 as _  # noqa: F401

        self.regs: dict[int, int] = {}
        self.activations: list[int] = [0] * 8
        self.weight_words: list[int] = [0] * 8
        self.outputs: list[int] = [0] * 8
        self.busy = False
        self.done = False

    def write(self, offset: int, value: int) -> None:
        if offset == REG_CTRL and (value & 1):
            self.outputs = bitlinear_gemv_int8(self.activations, self.weight_words)
            self.busy = False
            self.done = True
        elif offset == REG_ACT:
            addr = (value >> 8) & 0x7
            raw = value & 0xFF
            data = raw if raw < 128 else raw - 256
            self.activations[addr] = data
        elif offset == REG_WGT:
            addr = (value >> 16) & 0x7
            self.weight_words[addr] = value & 0xFFFF
        elif offset == REG_OUTIDX:
            self._out_idx = value & 0x7
        self.regs[offset] = value

    def read(self, offset: int) -> int:
        if offset == REG_STATUS:
            return (1 if self.busy else 0) << 1 | (1 if self.done else 0)
        if offset == REG_OUTDAT:
            idx = getattr(self, "_out_idx", 0)
            v = self.outputs[idx]
            if v < 0:
                return v & 0xFFFFFFFF
            return v
        return self.regs.get(offset, 0)


def run_system_vector() -> bool:
    vec = load_vector(ROOT / "tools" / "golden" / "vectors" / "system_8x8.json")
    mmio = MockMmio()
    for i, a in enumerate(vec["activations"]):
        mmio.write(REG_ACT, (i << 8) | (int(a) & 0xFF))
    for i, h in enumerate(vec["weight_words_hex"]):
        mmio.write(REG_WGT, (i << 16) | int(h, 16))
    mmio.write(REG_CTRL, 1)
    expected = vec["expected_outputs"]
    for i in range(8):
        mmio.write(REG_OUTIDX, i)
        got = mmio.read(REG_OUTDAT)
        if got >= 0x80000000:
            got -= 0x100000000
        if got != expected[i]:
            print(f"OUT[{i}] got {got} expected {expected[i]}")
            return False
    print("PASS: MMIO flow matches system_8x8.json")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run_system_vector() else 1)
