# BitNet b1.58 on a Zybo Z7-10

Running Microsoft's **BitNet b1.58 2B-4T** on a Digilent Zybo Z7-10 (Zynq-7010,
1 GB DDR3L), with the ternary matrix work in the FPGA fabric and everything else
on the ARM cores. Target: **≥ 2 tokens/s**.

Design and reasoning: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Where things stand: [`STATUS.md`](STATUS.md).

## Layout

| Path | What |
|---|---|
| `rtl/` | Streaming GEMV engine (Verilog) and its testbenches |
| `fpga/scripts/` | Vivado block design / bitstream, Vitis app build, JTAG program and run |
| `fpga/ps/` | Bare-metal PS applications |
| `tools/bitnet/` | Python reference model (golden), weight packer, generation CLI |
| `tools/setup/` | Toolchain environment, FTDI driver check, bitnet.cpp build + patches |
| `docs/` | Architecture, JTAG troubleshooting, PC reference inference, results |

The original 8×8 capstone RTL lives in its own repo,
[CapstoneBitnet/bitnet-fpga-accelerator](https://github.com/CapstoneBitnet/bitnet-fpga-accelerator)
(checked out locally in `bitnet-fpga-accelerator/`, ignored here). The design
that used it is preserved at tag `baseline-8x8-uart`.

## Setup

* Vivado/Vitis 2026.1 in `D:\Xilinx`; source `tools/setup/env.ps1` in each new shell.
* Zybo board files: `D:\xilinx-tools\vivado-boards\new\board_files`.
* Model: `C:\AI\models\BitNet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf` and the
  tokenizer in `C:\AI\models\BitNet-b1.58-2B-4T\`.
* Python 3.13 with `numpy`, `tokenizers`, `pyserial`; `gguf-py` comes from the
  bitnet.cpp clone (`BITNET_CPP_ROOT`, default `D:\src\BitNet`).

Run `make help` for the command list.

## Board facts

| | |
|---|---|
| Part | `xc7z010clg400-1` |
| Board part | `digilentinc.com:zybo-z7-10:part0:1.2` |
| UART | COM4 on this PC |
| Engine CSR base | `0x43C00000` |

Before any JTAG command, `make check-ftdi` must pass. Mismatched FTDI drivers
crashed Windows twice (see `docs/ZYBO_JTAG.md`).
