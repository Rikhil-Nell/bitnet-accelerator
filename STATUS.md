# Status

Target: BitNet b1.58 2B-4T on the Zybo Z7-10, as fast as the board allows (stretch: 10 tok/s).
Design: `docs/ARCHITECTURE.md`. Results: `docs/results/`.

## Run it

0. **Power the board from a 5 V supply** (barrel jack, 2.1 mm centre-positive, ≥ 2.5 A; jumper
   JP6 on WALL). At full DDR streaming the design draws more than a laptop USB 2 port supplies:
   the board browns out and resets (FPGA DONE drops, the UART disappears). Vivado estimates
   1.84 W on-chip (PS7 1.42 W), plus DDR3L, regulators and the rest of the board.
1. SD card (FAT32 or exFAT; superfloppy or partition 1) with `BITNET.BIN` (v2, 782 MiB)
   in the root: `make pack` writes it to `$(BITNET_WORK)/BITNET.BIN`. Board powered, USB
   connected.
2. `make board`: programs the FPGA, starts both ARM cores, loads the weights (~45 s from
   SD, ~3 s if they are still in DDR from the previous run), prints `READY`.
3. `make chat` (terminal) or `make demo` (web page with the write-up, chat and live
   profile at http://127.0.0.1:8000).

Firmware commands over the UART (`python tools/bitnet/board_cmd.py "<cmd>"`): `F <div>` engine
clock (1000 MHz / div; boots at 12 = 83 MHz, 7 = 142.9 MHz needs a 5 V supply), and toggles
`D` second core, `N` speculative guessing (off = same pass without guesses, to check exactness),
`C` top-5 candidate trace.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 0 | Repo cleanup, baseline tagged | done |
| 1 | Engine v1: 4 units, 2-bit ternary + int8 | done: 3.1–3.4 tok/s (tag `board-v1`) |
| 2 | NEON kernels, coarse ternary LM head, profiler | done: 5.1 tok/s |
| 3 | Engine v2: 1.6-bit T5 weights, progress counters; v2 image | done (sim + hardware bench) |
| 4 | Streamed forward pass (epilogues behind the engine) | done |
| 5 | Dual-core attention | done |
| 6 | Engine v3: two vectors per pass; speculative decoding; two-token prefill | done |
| 7 | Web write-up + live demo | done (`make demo`) |
| 8 | Standalone boot (BOOT.BIN), Linux | not started |

## Tools

* `make sim STALL=n`: engine RTL vs NumPy vectors, random AXI stalls.
* `make bench`: engine on hardware vs the A9 (T5, int8, two vectors, progress counters),
  DDR bandwidth by slice offset, one decode step's matrix work.
* `tools/bitnet/gpu_model.py`: the board model on a CUDA GPU (exact int8 matmuls),
  batched; used for corpora, draft tables and quick checks of new image formats.
* `tools/bitnet/eval_coarse_head.py`, `eval_spec.py`, `ngram_table.py`: the experiments
  behind the coarse head and speculative decoding.
