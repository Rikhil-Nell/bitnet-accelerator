"""Open the board's UART with an explanation instead of a traceback when it fails.

Only one program can hold a COM port: the web demo (web/server.py), a chat, uart_log or
a serial terminal. They all go through open_port().
"""
from __future__ import annotations

import sys


def open_port(port: str, baud: int = 115200, timeout: float = 0.5):
    import serial
    from serial.tools import list_ports

    try:
        return serial.Serial(port, baud, timeout=timeout)
    except serial.SerialException as e:
        ports = ", ".join(f"{p.device} ({p.description})" for p in list_ports.comports()) or "none"
        if "PermissionError" in str(e) or "Access is denied" in str(e):
            msg = (f"{port} is in use by another program. Only one can talk to the board at a time:\n"
                   f"  close the web demo (make demo / web/server.py), any other chat, uart_log\n"
                   f"  or serial terminal, then try again.")
        else:
            msg = (f"cannot open {port}: {e}\n  serial ports on this PC: {ports}\n"
                   f"  (set COM=... on the make command line or in local.mk)")
        print(f"error: {msg}", file=sys.stderr)
        raise SystemExit(2)
