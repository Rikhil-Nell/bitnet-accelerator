# Capstone status (evidence-based)

Last updated: 2026-11-01

**Board test PASS on Zybo Z7-10:** UART `PASS: system_8x8 on PL accelerator` (`build/capstone/uart.log`).

Quick audit (no rebuild): `make capstone-check` (`build/capstone/matrix.log`)

## Command matrix

| Command | Status | Evidence |
|---------|--------|----------|
| `make eval` | **PASS** | Golden, kernel, MMIO mock, I2_S remap (`build/capstone/eval.log`) |
| `make sim-all` | **PASS** | Vivado xsim fallback (no Icarus): decoder, MAC, accelerator, and AXI-level `bitnet_ip_top_tb` (`build/sim/xsim*.log`) |
| `make vivado-synth` | **PASS** | `build/vivado/utilization.rpt`, `timing.rpt` |
| `make bitstream` | **PASS** | `build/vivado/bitnet_zybo.bit`, `.xsa`; routed WNS +12.55 ns |
| `make check-ftdi` | **PASS** | `ftdibus.sys` and `ftser2k.sys` both at 2.12.36.20 (guards `list-hw`/`program`/`run-ps`) |
| `make list-hw` | **PASS** | Target `Digilent/210351AB6EECA`; devices `arm_dap_0 xc7z010_1` |
| `make program` | **PASS** | `Programmed D:/xilinx-tools/bitnet_zybo.bit to xc7z010_1` |
| `make ps-app` / `ps-app-rebuild` | **PASS** | `system_8x8_test.elf` (STATUS polling) |
| `make run-ps` | **PASS** | XSDB: `ps7_init`, `ps7_post_config`, `dow`, `con` |
| `make board-test` | **PASS** | UART on COM4: `PASS: system_8x8 on PL accelerator` |
| `make test-bitnet-smoke` | **SKIP** | No I2_S GGUF on disk; I2_S remap itself verified by `make test-i2s` |

## Board bring-up order

1. `make check-ftdi`
2. `make program`
3. `make uart-listen COM=COM4` (start first; PASS line is printed once)
4. `make run-ps`

## Fixes found during bring-up (2026-11-01)

- **Windows BSODs (`0x139`, `0x50`)** in `ftdibus.sys` from `hw_server.exe`: FTDI bus/port driver version mismatch. Fixed by installing CDM 2.12.36.20 for both and deleting the 2017 packages. See `docs/ZYBO_JTAG.md`.
- **`program_zybo.tcl`** picked `arm_dap_0`; it now selects `xc7z*`.
- **A9 hang on first AXI access**: `run_ps_zybo.tcl` lacked `ps7_post_config` (PS-PL level shifters).
- **STATUS.done never seen by software**: the FSM `done` is a 1-cycle pulse; `bitnet_ip_top` now latches it until the next start (matches `mmio_runner.py`). Covered by `bitnet_ip_top_tb`.
- **`run_ps_xsdb.ps1`** uses the official `xsdb.bat` launcher with a minimal PATH.

## BitNet 2B inference (capstone objective)

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Model weights on disk | **DONE** | `C:\AI\models\BitNet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf` |
| 2 | Build bitnet.cpp + reference reply | **DONE** | `D:\src\BitNet\build\bin\Release\llama-cli.exe`; `build/bitnet/bitnet_cpp_france_nocnv.log` |
| 3 | Real shapes via 8×8 tiling | **DONE** | `make bitnet-test-tiling`, `make bitnet-test-fused`; `board_check` 640×2560 on PL |
| 4 | Full model BitLinear on Zybo + generated reply | **DONE** | `build/bitnet/board_france.json`: backend `zybo-pl`, 30 layers, prefill ~5844 s + 1 token ~519 s; output **"The"** (id 791) to France prompt; log `build/bitnet/board_france.log` |

| Artifact | Notes |
|----------|--------|
| Emulated Paris reply | `build/bitnet/emulated_france.json` |
| Weights in DDR | `D:/xilinx-tools/bitnet_weights.bin`; load log `build/bitnet/load_weights.log` |
| Firmware | `fpga/ps/bitlinear_server/` (v3 fused FFN in source; board may still run v2 until `make bitlinear-ps-app` + `make run-bitlinear-server`) |
| Docs | `docs/BITNET_BOARD_INFERENCE.md`, `docs/BITNET_PERFORMANCE.md` |

## Optional next

- **bitnet.cpp reference**: `docs/BITNET_PC_INFERENCE.md` (VS Clang + CMake).
- **Icarus**: `choco install iverilog -y` for faster local sim.
