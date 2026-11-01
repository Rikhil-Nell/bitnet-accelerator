# Status

Target: BitNet b1.58 2B-4T at **≥ 2 tok/s** on the Zybo Z7-10. Design: `docs/ARCHITECTURE.md`.

## Baseline (tag `baseline-8x8-uart`)

The first design used a serial 1-MAC/cycle core fed by the CPU over AXI-Lite, with the PC
driving every matmul over UART. It produced one correct token ("The") in about 106 minutes:
5844 s of prefill and 519 s/token. That is about 1000× short of the target, so it was
replaced rather than tuned. Results are in `docs/results/baseline-8x8-uart/`.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 0 | Repo cleanup, baseline tagged, architecture written | done |
| 1 | GEMV engine RTL (ternary + int8), xsim against Python vectors | done: bit-exact at 0/25/60 % AXI stalls, 0.98–0.99 beats/clk/unit |
| 2 | Block design: 4 HP ports at ~150 MHz, bitstream, timing met | |
| 3 | Board: random-matrix correctness + measured DDR bandwidth | |
| 4 | Weight packer (engine format, int8 LM head) + SD card loader | |
| 5 | Bare-metal forward pass on the A9, layer-by-layer vs `engine.py` | |
| 6 | PC chat client over UART; tokens/s measured | |
| 7 | Headroom: 1.6-bit packing, second core, Linux | |

## Known-good infrastructure

- FTDI driver mismatch fixed; `make check-ftdi` guards every JTAG target (`docs/ZYBO_JTAG.md`).
- JTAG program and XSDB run flow, including `ps7_post_config`.
- Reference: `make generate-emulated` answers "The capital of France is Paris."; bitnet.cpp
  builds with the patches in `tools/setup/bitnet-cpp-patches/`.
