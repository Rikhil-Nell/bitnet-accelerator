# Status

Target: BitNet b1.58 2B-4T at **≥ 2 tok/s** on the Zybo Z7-10. Design: `docs/ARCHITECTURE.md`.

**Current: 3.1–3.4 tok/s decode, ~0.2 s per prompt token, correct answers.**
Details: `docs/results/board-v1/README.md`.

## Run it

1. SD card (FAT32 or exFAT; superfloppy or partition 1) with `BITNET.BIN` in the root
   (`make pack` writes it to `D:\xilinx-tools\BITNET.BIN`). Board powered, USB connected.
2. `make board`: programs the FPGA, boots the firmware, loads the weights (~45 s to `READY`).
3. `make chat`: ask questions. `/reset` clears the context, `/quit` exits;
   `python tools/bitnet/chat.py --multi` keeps the conversation (512-token context).

## Milestones

| # | Milestone | Status |
|---|---|---|
| 0 | Repo cleanup, baseline tagged, architecture written | done |
| 1 | GEMV engine RTL (ternary + int8), xsim against Python vectors | done: bit-exact at 0–60 % AXI stalls |
| 2 | Block design: 4 HP ports, bitstream, timing met | done: 142.857 MHz, +0.43 ns, 28 % LUT |
| 3 | Board: random-matrix correctness + measured DDR bandwidth | done: bit-exact, ~4.0 GB/s (95 % of peak) |
| 4 | Weight packer (engine format, int8 LM head) + SD card loader | done: 812 MiB image, 41 s load, checksum |
| 5 | Bare-metal forward pass on the A9 | done: correct answers |
| 6 | PC chat client over UART; tokens/s measured | done: 3.1–3.4 tok/s |
| 7 | Headroom (below) | next |

## Where the remaining time goes (per token, ~300 ms)

- ~219 ms PL matrix work (DDR-bound at ~95 % of peak)
- the rest on the A9: attention over the KV cache (grows with context), norms,
  activation quantization, cache maintenance, argmax over 128K logits

## Next

- Overlap A9 work with PL streaming and use the second A9 core for attention.
- Batched prefill: one weight pass for several prompt tokens (prompt is ~0.2 s/token now).
- 1.6-bit ternary packing (−20 % bytes → ~+20 % tok/s on the matrix part).
- Boot from SD without the PC (BOOT.BIN), then optionally Linux on the PS.

## Baseline (tag `baseline-8x8-uart`)

Serial 1-MAC core fed over AXI-Lite, PC driving every matmul over UART: one token
("The") after ~106 minutes. Results in `docs/results/baseline-8x8-uart/`.
