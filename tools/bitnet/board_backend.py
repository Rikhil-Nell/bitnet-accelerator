#!/usr/bin/env python3
"""BitLinear backend that runs every ternary matmul on the Zybo PL accelerator.

Talks to fpga/ps/bitlinear_server over UART. Weights must already be in board
DDR (tools/bitnet/pack_weights.py + make load-weights).
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np

from engine import EmulatedAccelerator
from gguf_model import BitNetModel

MAGIC = b"\xB1\x7A"
BAUD = 921600
MANIFEST = Path("D:/xilinx-tools/bitnet_weights.json")
MAX_T = 64
GT_HZ = 666_666_687 / 2


class BoardError(RuntimeError):
    pass


class BoardLink:
    def __init__(self, port: str = "COM4", baud: int = BAUD, timeout: float = 300.0):
        import serial

        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.ser.reset_input_buffer()
        self._io_timeout = timeout

    def _read(self, n: int) -> bytes:
        data = self.ser.read(n)
        if len(data) != n:
            raise BoardError(f"timeout: wanted {n} bytes, got {len(data)}")
        return data

    def _expect_ok(self) -> None:
        hdr = self._read(2)
        if hdr == b"ER":
            (code,) = struct.unpack("<I", self._read(4))
            raise BoardError(f"board error code {code}")
        if hdr != b"OK":
            raise BoardError(f"bad response header {hdr!r}")

    def ping(self) -> int:
        self.ser.timeout = min(self._io_timeout, 5.0)
        self.ser.write(MAGIC + b"P")
        self._expect_ok()
        v = struct.unpack("<I", self._read(4))[0]
        self.ser.timeout = self._io_timeout
        return v

    def self_test(self) -> bool:
        self.ser.write(MAGIC + b"S")
        self._expect_ok()
        return struct.unpack("<I", self._read(4))[0] == 1

    def checksum(self, addr: int, nbytes: int) -> int:
        self.ser.write(MAGIC + b"C" + struct.pack("<II", addr, nbytes))
        self._expect_ok()
        return struct.unpack("<I", self._read(4))[0]

    def gemm(
        self, mats: list[tuple[int, int]], xq: np.ndarray, *, checksum_out: bool = True
    ) -> tuple[np.ndarray, float]:
        """mats = [(addr, n), ...] sharing input xq (T, K) int8 -> (T, sum n) int32, seconds on board."""
        t, k = xq.shape
        req = MAGIC + b"G" + struct.pack("<BHH", len(mats), k, t)
        for addr, n in mats:
            req += struct.pack("<IH", addr, n)
        xb = np.ascontiguousarray(xq, dtype=np.int8).tobytes()
        req += xb + struct.pack("<I", sum(xb) & 0xFFFFFFFF)
        self.ser.write(req)
        self._expect_ok()
        (ticks,) = struct.unpack("<I", self._read(4))
        total_n = sum(n for _, n in mats)
        raw = self._read(4 * t * total_n)
        y = np.frombuffer(raw, dtype="<i4").reshape(t, total_n)
        (ysum,) = struct.unpack("<I", self._read(4))
        if checksum_out and int(y.view("<u4").sum(dtype=np.uint64) & 0xFFFFFFFF) != ysum:
            raise BoardError("output checksum mismatch")
        return y.copy(), ticks / GT_HZ


class BoardAccelerator:
    """Same interface as EmulatedAccelerator; optionally cross-checks every result."""

    name = "zybo-pl"

    def __init__(
        self,
        model: BitNetModel,
        port: str = "COM4",
        verify: bool = True,
        check_weights: bool = True,
        fuse_ffn: bool = True,
        checksum_out: bool = True,
    ):
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["matrices"]
        self.link = BoardLink(port)
        print(f"[board] firmware v{self.link.ping()}  self-test={'PASS' if self.link.self_test() else 'FAIL'}",
              flush=True)
        self.ref = EmulatedAccelerator(model) if verify else None
        self.calls = 0
        self.macs = 0
        self.board_seconds = 0.0
        self.mismatches = 0
        self.fuse_ffn = fuse_ffn
        self.checksum_out = checksum_out
        if check_weights:
            self.check_weights(model.n_layer)

    def check_weights(self, n_layer: int) -> None:
        t0 = time.perf_counter()
        bad = []
        for key, m in self.manifest.items():
            if int(key.split(".")[0]) >= n_layer:
                continue
            if self.link.checksum(m["addr"], m["bytes"]) != m["checksum"]:
                bad.append(key)
        if bad:
            raise BoardError(f"weights in board DDR do not match for {len(bad)} matrices, e.g. {bad[:3]}")
        print(f"[board] weight checksums OK ({time.perf_counter() - t0:.1f}s)", flush=True)

    def _gemm_manifests(self, specs: list[tuple[int, str]], xq: np.ndarray) -> np.ndarray:
        mats = [(self.manifest[f"{layer}.{proj}"]["addr"], self.manifest[f"{layer}.{proj}"]["n"]) for layer, proj in specs]
        outs = []
        for s in range(0, xq.shape[0], MAX_T):
            chunk = xq[s : s + MAX_T]
            y, secs = self.link.gemm(mats, chunk, checksum_out=self.checksum_out)
            self.board_seconds += secs
            outs.append(y)
        return np.concatenate(outs, axis=0)

    def matmul(self, layer: int, proj: str, xq: np.ndarray) -> np.ndarray:
        m = self.manifest[f"{layer}.{proj}"]
        outs = []
        for s in range(0, xq.shape[0], MAX_T):
            y, secs = self.link.gemm(
                [(m["addr"], m["n"])], xq[s : s + MAX_T], checksum_out=self.checksum_out
            )
            self.board_seconds += secs
            outs.append(y)
        y = np.concatenate(outs, axis=0)
        self.calls += 1
        self.macs += xq.shape[0] * m["n"] * m["k"]
        if self.ref is not None:
            expect = self.ref.matmul(layer, proj, xq)
            if not np.array_equal(y, expect):
                self.mismatches += 1
                bad = int(np.sum(y != expect))
                raise BoardError(f"layer {layer} {proj}: {bad} outputs differ from emulation")
        return y

    def matmul_fused(self, layer: int, projs: tuple[str, ...], xq: np.ndarray) -> list[np.ndarray]:
        """One UART round-trip for several matrices with the same K (e.g. ffn_gate + ffn_up)."""
        sizes = [self.manifest[f"{layer}.{p}"]["n"] for p in projs]
        y_all = self._gemm_manifests([(layer, p) for p in projs], xq)
        self.calls += 1
        self.macs += xq.shape[0] * sum(s * self.manifest[f"{layer}.{projs[0]}"]["k"] for s in sizes)
        parts = np.split(y_all, np.cumsum(sizes)[:-1], axis=1)
        if self.ref is not None:
            for p, y in zip(projs, parts):
                expect = self.ref.matmul(layer, p, xq)
                if not np.array_equal(y, expect):
                    raise BoardError(f"layer {layer} {p}: fused output differs from emulation")
        return [p.copy() for p in parts]
