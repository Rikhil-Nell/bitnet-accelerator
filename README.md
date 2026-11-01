# BitNet b1.58 on a Zybo Z7-10

Running Microsoft's **BitNet b1.58 2B-4T** on a Digilent Zybo Z7-10 (Zynq-7010,
1 GB DDR3L), with the ternary matrix work in the FPGA fabric and everything else
on the two ARM cores. Measured: **8.25 tokens/s** on USB power with the engine at
100 MHz; about 10 tokens/s projected at the full 142.9 MHz clock on a 5 V supply
(`docs/results/board-v3/`).

New machine: [`SETUP.md`](SETUP.md). Once set up: SD card with `BITNET.BIN` in the
board, then `make board` and `make chat` (or `make demo` for the web page).

Design and reasoning: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Where things stand: [`STATUS.md`](STATUS.md). `make help` lists every command.

## Layout

| Path | What |
|---|---|
| `rtl/` | Streaming GEMV engine (Verilog) and its testbenches |
| `fpga/scripts/` | Vivado block design / bitstream, Vitis app build, JTAG program and run |
| `fpga/ps/` | Bare-metal firmware (`bitnet_zybo`) and the engine test (`gemv_bench`) |
| `tools/bitnet/` | Reference models (NumPy, CUDA), weight packer, chat client, experiments |
| `tools/setup/` | Toolchain environment, FTDI driver check, bitnet.cpp build + patches |
| `web/` | Web page: write-up for non-specialists + live chat/profile (`make demo`) |
| `docs/` | Architecture, JTAG troubleshooting, PC reference inference, results |

The original 8×8 capstone RTL lives in its own repo,
[CapstoneBitnet/bitnet-fpga-accelerator](https://github.com/CapstoneBitnet/bitnet-fpga-accelerator).
The design that used it is preserved at tag `baseline-8x8-uart`; this repo no
longer depends on it.

## Board facts

| | |
|---|---|
| Part | `xc7z010clg400-1` |
| Board part | `digilentinc.com:zybo-z7-10:part0:1.2` |
| Engine CSR base | `0x43C00000` |
| Power | 5 V barrel jack (J17, JP6 on WALL) for full speed; USB power limits the engine clock |

Before any JTAG command, `make check-ftdi` must pass. Mismatched FTDI drivers
crashed Windows twice (see `docs/ZYBO_JTAG.md`).
