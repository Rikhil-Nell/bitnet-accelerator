# Setup on a new Windows machine

Everything below is needed once per laptop. Paths are examples: put your own in
`local.mk` (step 6) and every script follows them.

## Hardware

* Digilent **Zybo Z7-10**, its micro-USB cable (programming + serial), a microSD card
  (≥ 1 GB, FAT32 or exFAT).
* **A 5 V supply into the barrel jack J17** (5.5 × 2.1 mm, centre positive, ≥ 2.5 A; a
  phone charger with a USB-A → 5.5 × 2.1 mm cable works), jumper **JP6 on WALL**. On USB power
  alone the board browns out above ~3.2 GB/s of memory streaming, so the firmware then has to
  run the engine at 83 MHz instead of 142.9 MHz.

## Software

1. **Git, Python 3.11+, Node.js 20+** (Node only to build the web page).
2. **AMD Vivado + Vitis 2026.1** (unified installer): select Vivado, Vitis Embedded, and
   Zynq-7000 device support. Let it install the cable drivers. Default location in this repo
   is `D:\Xilinx`; anything else goes in `local.mk`.
3. **Digilent board files**:
   `git clone https://github.com/Digilent/vivado-boards C:/xilinx-tools/vivado-boards`
   (the scripts use `<clone>/new/board_files`; set `BITNET_BOARD_FILES` if it lives elsewhere).
4. **bitnet.cpp** (only its Python GGUF reader is used, no build needed):
   `git clone --recursive https://github.com/microsoft/BitNet.git C:/src/BitNet`
5. **Python packages and the model**:

   ```
   pip install -r requirements.txt
   huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir C:/models/BitNet-b1.58-2B-4T-gguf
   huggingface-cli download microsoft/BitNet-b1.58-2B-4T --local-dir C:/models/BitNet-b1.58-2B-4T
   ```

6. **Per-machine paths**: copy `local.mk.example` to `local.mk` and edit it (Xilinx location,
   a work directory with ~3 GB free for the board image and Vitis workspace, model paths, and
   the board's COM port from Device Manager → Ports).

## Build (first time ~45 minutes, mostly Vivado)

```
make check-ftdi                   # FTDI driver versions must match (docs/ZYBO_JTAG.md)
make sim                          # engine RTL vs NumPy, ~2 min
make bitstream                    # ~20 min -> build/vivado/bitnet_zybo.bit + .xsa
make ps-app APP=bitnet_zybo       # Vitis platform + firmware
make ps-app APP=gemv_bench        # optional: engine test program
make pack                         # ~1 min -> $(BITNET_WORK)/BITNET.BIN (782 MiB)
make web                          # web page (npm install + build)
```

Copy `BITNET.BIN` to the root of the SD card and put the card in the Zybo. (Instead of
`make pack` you can copy a teammate's `BITNET.BIN`; at boot the firmware prints its checksum,
`69DC1E2E` for the current image.)

## Run

```
make board                        # USB power: engine at 83 MHz
make board FCLK_DIV=7             # 5 V supply: engine at 142.9 MHz
make chat                         # terminal chat (python tools/bitnet/chat.py --multi keeps context)
make demo                         # web page with write-up + live chat at http://127.0.0.1:8000
python web/server.py --host 0.0.0.0   # let other devices on the network open the page
make bench                        # optional: engine checks and bandwidth on the board
```

If `make board` stops with "Could not select Cortex-A9 #0", run `make srst` once, then
`make board` again (the weights reload from the SD card in ~45 s).

## What is generated and where

| Artifact | Made by | Location |
|---|---|---|
| bitstream, XSA, ps7_init.tcl | `make bitstream` | `build/vivado/` |
| firmware ELFs | `make ps-app` | `$(BITNET_WORK)/vitis_ws/<app>/build/` |
| weight image | `make pack` | `$(BITNET_WORK)/BITNET.BIN` |
| draft table (committed) | `tools/bitnet/ngram_table.py` | `fpga/ps/bitnet_zybo/ngram_table.h` |

Regenerating the draft table needs a CUDA GPU and PyTorch (`tools/bitnet/spec_corpus2.py`,
then `ngram_table.py`); the committed one works as is.
