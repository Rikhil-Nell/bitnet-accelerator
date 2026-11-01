#!/usr/bin/env python3
"""Print (and optionally save) board UART output until a line matches --until or the timeout.

  python tools/bitnet/uart_log.py --until "BENCH (PASS|FAIL)" --timeout 600 --out build/board/bench.log
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--until", default=None, help="regex that ends the capture")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import serial

    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    ser.reset_input_buffer()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    out = args.out.open("w", encoding="utf-8") if args.out else None
    pat = re.compile(args.until) if args.until else None
    end = time.time() + args.timeout
    print(f"[uart_log] listening on {args.port} @ {args.baud}", flush=True)
    while time.time() < end:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode(errors="replace").rstrip()
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()
        if pat and pat.search(line):
            return 0
    print("[uart_log] timeout", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
