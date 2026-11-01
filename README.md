# BitNet Zybo accelerator (capstone)

FPGA **W2A8 BitLinear** engine for **Digilent Zybo Z7-10** (`xc7z010`), with PC-side golden kernel and Zynq bring-up.

## Quick commands

Source Vivado once per shell:

```powershell
. tools/setup/env.ps1
```

| Target | Purpose |
|--------|---------|
| `make capstone-check` | Status matrix (host + artifacts; JTAG/board) |
| `make eval` | Golden L0 tests |
| `make test-kernel` | PC BitLinear == golden (+ NumPy reference) |
| `make test-mmio` | CSR/MMIO flow for `system_8x8.json` |
| `make vivado-synth` | RTL-only synth sanity (no BD) |
| `make bitstream` | Zynq BD + `build/vivado/bitnet_zybo.bit` |
| `make ps-app` | Vitis bare-metal `system_8x8_test.elf` |
| `make ps-app-rebuild` | Sync PS sources + rebuild (platform already built) |
| `make list-hw` | JTAG diagnostic |
| `make program` | JTAG program `.bit` (`docs/ZYBO_JTAG.md`) |
| `make run-ps` | XSDB download/run `system_8x8_test.elf` |
| `make board-bringup COM=COMx` | Program PL + run PS + UART instructions |
| `make test-bitnet-smoke` | Microsoft **bitnet.cpp** `run_inference.py` (if built) |
| `make test-bitnet-layer` | Capstone kernel vs HF weight slice JSON |
| `make sim-all` | RTL TB smoke (Icarus if in PATH, else Vivado **xsim**) |
| `make uart-listen COM=COMx` | Board UART PASS line (115200) |

Optional faster RTL sim: `choco install iverilog -y` (Admin). JTAG stuck? `. tools/setup/diagnose_zybo_usb.ps1` (Converter B `CM_PROB_NEED_RESTART` → reboot). Then `. tools/setup/board_bringup.ps1 -Com COMx`.

## Microsoft BitNet inference (PC)

Official engine: **[microsoft/BitNet](https://github.com/microsoft/BitNet)** → `ggml-model-i2_s.gguf` + `run_inference.py`.

- Guide: **`docs/BITNET_PC_INFERENCE.md`**
- Helper: `. tools/setup/bitnet-pc.ps1`
- Env: `BITNET_CPP_ROOT`, `BITNET_MODEL_DIR`, `BITNET_GGUF`
- On this machine, **`C:\AI\models`** has gemma/qwen/parakeet — **BitNet weights not present yet** (download commands in the doc).

Extract an 8×8 FPGA-aligned slice from HF safetensors:

```powershell
python tools/host/extract_bitlinear_slice.py
make test-bitnet-layer
```

## Docs

- `docs/DESIGN.md` — architecture, encoding, phases  
- `docs/ITERATION.md` — current step  
- `docs/CAPSTONE_BRINGUP.md` — board checklist after JTAG fix  
- `docs/BITNET_PC_INFERENCE.md` — bitnet.cpp + weights  
- `tools/setup/README.md` — Vivado 2026.1 on `D:\Xilinx`  

## Board

- Part: `xc7z010clg400-1`  
- Board part: `digilentinc.com:zybo-z7-10:part0:1.2`  
- Board files: `D:\xilinx-tools\vivado-boards\new\board_files`  
- Accelerator AXI base: **`0x43C00000`** (see `docs/ITERATION.md`)
