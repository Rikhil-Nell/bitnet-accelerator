"""Sync fpga/ps/$BITNET_PS_APP sources and rebuild that existing app component (platform unchanged)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["VITIS_EMBEDDED_INSTALL"] = "true"

REPO = Path(__file__).resolve().parents[2]
WS = Path(os.environ.get("BITNET_VITIS_WS", "D:/xilinx-tools/vitis_ws"))
APP = os.environ.get("BITNET_PS_APP", "gemv_bench")
PS_SRC = REPO / "fpga/ps" / APP

import vitis

app_src = WS / APP / "src"
app_src.mkdir(parents=True, exist_ok=True)
for src in PS_SRC.iterdir():
    if src.is_file():
        (app_src / src.name).write_bytes(src.read_bytes())

ide = WS / "_ide"
for stale in (ide / ".peers.ini", ide / "workspace_journal.py"):
    if stale.is_file():
        stale.unlink(missing_ok=True)

client = vitis.create_client(workspace=str(WS))
comp = client.get_component(name=APP)
rc = comp.build()
vitis.dispose()
print(f"build rc={rc}")
raise SystemExit(0 if rc == 0 else 1)
