# BitNet on Zybo (BitLinear on PL)

Host runs attention / norms / LM head; **ternary matmuls** go to `fpga/ps/bitlinear_server` over UART at **921600** baud.

## One-time / after code changes

```powershell
make program
make bitlinear-ps-app          # Vitis app: fpga/ps/bitlinear_server
make pack-bitnet-weights       # -> D:/xilinx-tools/bitnet_weights.bin + .json (~497 MiB)
```

## Load weights (JTAG, slow)

```powershell
make load-bitnet-weights       # log: build/bitnet/load_weights.log
```

Re-use DDR contents on the same power session with `BITNET_SKIP_LOAD=1 make run-bitlinear-server`.

## Run server + check

Start the server **before** any host tool (`board_check`, `generate --backend board`). The server UART is **921600** (not 115200).

```powershell
make program                   # after power-cycle
make run-bitlinear-server      # JTAG: load weights + start bitlinear_server.elf
python tools/bitnet/board_check.py --port COM4 --layer 0 --proj attn_k
```

If `board_check` times out on ping, the A9 firmware is not running yet (re-run `make run-bitlinear-server` or check COM4).

## Generate (PC reference vs board)

```powershell
python tools/bitnet/generate.py -p "What is the capital of France?" -n 20 --backend emulated
python tools/bitnet/generate.py -p "..." -n 5 --backend board --port COM4
```

Expect **minutes per token** on the first 8×8 MMIO design unless you use optimized firmware (v3+) and `generate.py --fast`; see [BITNET_PERFORMANCE.md](BITNET_PERFORMANCE.md) for the path toward ~0.5 s/token.

## Microsoft bitnet.cpp reference (PC)

See `docs/BITNET_PC_INFERENCE.md`. Requires VS 2022 **Clang + CMake** components.
