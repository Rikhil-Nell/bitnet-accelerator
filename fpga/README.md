# FPGA / Zybo Z7-10 flow

Block design + bitstream for **PS7 + AXI GP0 + `bitnet_ip_top`** (8×8 BitLinear accelerator).

## Prerequisites

- Vivado **2026.1** at `D:\Xilinx\2026.1\`
- Board repo: `D:\xilinx-tools\vivado-boards\new\board_files`
- Board part: `digilentinc.com:zybo-z7-10:part0:1.2`

```powershell
. tools/setup/env.ps1
```

## Commands (repo root)

| Target | Output |
|--------|--------|
| `make vivado-synth` | RTL-only synth check on `bitnet_ip_top` |
| `make bitstream` | `build/vivado/bitnet_zybo.bit`, `build/vivado/bitnet_zybo.xsa` |
| `make program` | JTAG program `.bit` (board USB + hw_server) |
| `make run-ps` | XSDB `dow` + `con` for `system_8x8_test.elf` |
| `make board-bringup COM=COMx` | `list-hw` → `program` → `run-ps` (+ UART hint) |
| `make ps-app` | Vitis embedded Python build (`tools/setup/env-vitis.ps1`) |
| `make export-xsa` | Re-export `bitnet_zybo.xsa` from Vivado project |

Tcl entry: `fpga/scripts/build_zybo_bd.tcl`

**AXI base:** `0x43C00000` (4 KiB) — matches `tools/host/mmio_runner.py`.

## PS bare-metal test

Sources: `fpga/ps/system_8x8_test/` — replays `system_8x8.json` via MMIO, prints PASS/FAIL on UART (**115200**).

**Vitis 2026 notes:** `xsct` is disabled. Use `make ps-app` (stages XSA to `D:/xilinx-tools/bitnet_zybo.xsa`, workspace `D:/xilinx-tools/bitnet_vitis_ws`). Close Vitis IDE if workspace is locked. Requires embeddedsw repo (script adds `Vitis/data/embeddedsw` automatically).

After `make bitstream`:

1. In **Vitis Unified**, create a platform from `build/vivado/bitnet_zybo.xsa` (GUI fallback).
2. Create a **standalone** app for `ps7_cortexa9_0`, add `fpga/ps/system_8x8_test/*.c`.
3. `make program` then `make run-ps` (XSDB batch; do not use `xsct` on 2026.x), or Vitis IDE Run/Debug.
4. `make uart-listen COM=COMx` — expect `PASS: system_8x8 on PL accelerator`.

Register map: `docs/ITERATION.md`.
