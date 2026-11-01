#!/usr/bin/env python3
"""Read Zybo UART until PASS/FAIL line or timeout (115200 8N1)."""
from __future__ import annotations

import argparse
import sys
import time

PASS_MARK = "PASS: system_8x8 on PL accelerator"
FAIL_MARK = "FAIL:"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3", help="UART port (Device Manager → USB Serial A)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        print("pip install pyserial", file=sys.stderr)
        return 1

    deadline = time.time() + args.timeout
    print(f"Listening on {args.port} @ {args.baud} for up to {args.timeout}s...")
    with serial.Serial(args.port, args.baud, timeout=0.5) as ser:
        buf = ""
        while time.time() < deadline:
            chunk = ser.read(256).decode("utf-8", errors="replace")
            if chunk:
                print(chunk, end="", flush=True)
                buf += chunk
                if PASS_MARK in buf:
                    print("\nBOARD TEST PASS")
                    return 0
                if FAIL_MARK in buf:
                    print("\nBOARD TEST FAIL")
                    return 1
    print("\nTimeout waiting for UART banner (is PS app running?)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
