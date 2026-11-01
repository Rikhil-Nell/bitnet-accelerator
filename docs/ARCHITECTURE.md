# Architecture: BitNet b1.58 2B-4T on the Zybo Z7-10

Everything runs on the board: the FPGA (PL) streams the ternary weights through four
matrix-vector engines, the two Cortex-A9 cores (PS) do everything else. The PC (or the
web page, `web/`) only converts text to token ids and back.

## The one equation

Generating a token is a matrix-vector product against every weight, and each weight is
used once. The board is **memory-bandwidth bound**:

```
tokens/s ≈ sustained DDR read bandwidth / bytes read per token
```

| Per token | Params | v1 bytes | v2 bytes |
|---|---|---|---|
| 30 layers, 7 ternary matrices each | 2.084 B | 521 MB (2 bit) | 424 MB (T5, 1.6 bit + padding/skew) |
| LM head ranking | 328 M | 328 MB (int8) | 66 MB (T5 coarse copy) + 64 int8 rows |
| **Total** | | **849 MB** | **490 MB** |

DDR3L, 32-bit @ 1066 MT/s: 4.27 GB/s peak, **4.05 GB/s measured** (95 %). So one pass
over the weights takes ≥ 121 ms (8.3 passes/s). Speculative decoding gets more than
one token out of a pass.

## Block diagram

```
          ┌─────────── PS: Cortex-A9 ×2 (bare-metal, NEON) ───────────┐
          │ CPU0: control, norms, quantize, RoPE, epilogues, drafting,  │
          │       top-64 rescoring, UART; attention for KV heads 0-2    │
          │ CPU1: attention for KV heads 3-4 (cpu1.c, WFE/SEV mailbox)  │
          └──────┬────────────────────────┬────────────────────────────┘
          M_AXI_GP0 (CSRs)        S_AXI_HP0..HP3 (DDR, 64-bit AXI3 each)
                 │              ┌─────────┼─────────┬─────────┐
          ┌──────▼──────┐  ┌────▼────┐ ┌──▼──────┐ ┌▼────────┐ ┌▼────────┐
          │ CSRs        │─▶│ unit 0  │ │ unit 1  │ │ unit 2  │ │ unit 3  │
          │ (AXI-Lite)  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘
          └─────────────┘  each unit: AXI3 read master → 512-deep FIFO →
                           T5 decode → 2 × 40-lane adder trees → accumulators
                           → output FIFO → AXI3 write master (+ progress count)
```

* Four identical units, one per HP port; a matrix is split into four equal row
  ranges. 4 ports × 8 B × 142.857 MHz = 4.57 GB/s of port capacity, above what the DDR
  delivers, so the DDR is the only bottleneck.
* Fused matrices: `q|k|v` (3840×2560) and `gate|up` (13824×2560); a layer is 4 engine
  commands, a token 120 + the coarse head.

## Engine (rtl/, "GEM3")

Per command each unit: (1) loads the int8 activation vector(s) into a 5-bank BRAM
buffer (two sets for two vectors), (2) streams its rows of weights, one 64-bit beat
per clock, (3) writes one int32 per row (two rows per word; with two vectors one
`{y_b, y_a}` word per row).

**T5 weights (mode 0).** Five ternary weights per byte in base 3:
`byte = Σ (t_i + 1)·3^i`, i = 0..4 (3⁵ = 243 ≤ 256), so a beat carries 40 weights.
A generated 256-entry decoder (`rtl/t5_decode.v`) turns each byte into five 2-bit
codes (00 = 0, 01 = +1, 11 = −1); each lane then adds, subtracts or skips its int8
activation, and a pipelined 40 → 10 → 5 → 1 adder tree sums the beat. K is padded to a
multiple of 40 (6912 → 6920) with zero weights and zero activations.

**Int8 (mode 1).** 8 weights per beat on DSP multipliers (exact-head fallback, tests).

**Two vectors (NB = 1).** The same decoded weights feed a second tree for a second
activation vector, so one pass over the weights evaluates two token positions.

**Progress counters.** Each unit counts acknowledged write bursts (16 beats = 32 rows,
16 with two vectors). Rows it reports are already in DDR, so the A9 starts on them
while the engine streams the rest.

Timing: 150 MHz target met (+0.30 ns, GEM2); runs at 142.857 MHz (IO PLL / 7).
Pipeline: FIFO pop → register (BRAM/FIFO outputs) → T5 decode → products → sums of 4
→ sums of 2 → beat total → accumulate. Simulation (`make sim STALL=n`) checks every
case bit-exact against NumPy with random AXI stalls, including 4 KiB boundaries and
WLAST.

### CSRs (AXI-Lite, base `0x43C0_0000`)

| Offset | Name | Meaning |
|---|---|---|
| 0x00 | CTRL | bit0 start |
| 0x04 | STATUS | bit0 done, bit1 busy, bit2 AXI error, [7:4] unit done, [11:8] unit error |
| 0x08 | MODE | 0 = T5 ternary, 1 = int8 |
| 0x0C | X_BEATS | activation beats per vector (K padded / 8) |
| 0x10 | ROWS | rows per unit |
| 0x14 | X_ADDR | activation vector(s), 128 B aligned; second vector at + 8·X_BEATS |
| 0x18 | CYCLES | PL clocks of the last command |
| 0x1C | VERSION | `GEM3` (0x47454D33) |
| 0x20 + 8u | W_ADDR[u] | weight slice of unit u |
| 0x24 + 8u | Y_ADDR[u] | output slice of unit u |
| 0x40 + 4u | PROGRESS[u] | write bursts acknowledged in the current command |
| 0x50 | ROW_BEATS | beats per weight row |
| 0x54 | NB | 1 = two activation vectors |

## Weight image (BITNET.BIN, `tools/bitnet/board_image.py`)

Loaded from the SD card into DDR at 0x0800_0000 (782 MiB, checksum-verified; reused
without reloading if it is already in DDR). Each matrix is stored as four unit slices
`off + u·ustride`, `ustride = round_up(slice, 32 KiB) + 8 KiB`: when the four slices
started a multiple of 32 KiB apart, all units hit the same DDR bank at once and
bandwidth halved (2.25 vs 4.05 GB/s, measured). Contents: T5 layers, the T5 coarse
head with per-row scales, the int8 head (embedding table + exact rescoring), norms.

## A9 software (fpga/ps/bitnet_zybo)

* **Streamed forward pass.** While the engine streams matrix *i*, CPU0 processes the
  finished rows: dequantize, residual add, ReLU² gate, and the statistics of the next
  RMSNorm (sum of squares, absmax of `x·w`). `quantize(rmsnorm(x)·w) = quantize(x·w)`
  because the 1/rms factor cancels in absmax scaling, so after the engine finishes only
  one quantize pass remains before matrix *i+1* can start.
* **Coarse LM head.** The engine ranks all 128,256 rows with the T5 copy; CPU0 keeps a
  running top-64 as rows arrive, then rescores those 64 rows exactly in int8. Over 250
  decode steps the exact winner was always in the coarse top 16
  (`tools/bitnet/eval_coarse_head.py`).
* **Attention** keeps four query heads (one GQA group) per NEON vector through scores,
  softmax and the value sum; KV cache in fp16, contiguous per KV head. CPU1 handles two
  of the five KV-head groups (`cpu1.c`: own L1/TLB invalidate, SMP coherency, CPU0's
  MMU table; started by xsdb at `cpu1_entry`).
* **Speculative decoding.** Each pass feeds the current token and a guess for the next
  one (engine NB = 1). The guess comes from the conversation itself (the token that
  followed the last two tokens earlier) or from a table of the model's own most
  frequent continuations (`tools/bitnet/ngram_table.py`, built from answers generated by
  `tools/bitnet/gpu_model.py` on a GPU). The model's prediction always decides; a right
  guess yields two tokens per pass, a wrong one only costs the second position's A9
  work. Prompts are read two tokens per pass the same way.
* Cache maintenance: the engine's x buffer is a normal non-cacheable section (writes
  stream, no flush); the y buffer is L1-only (outer non-cacheable), so coherence needs
  only L1 line invalidates.

## Correctness ladder

1. `tools/bitnet/engine.py` (NumPy) and `tools/bitnet/gpu_model.py` (CUDA, exact int8
   matmuls) are the golden models.
2. RTL simulation against Python vectors, random AXI stalls.
3. On board (`make bench`): random T5 / int8 / two-vector matrices against the A9;
   rows reported by the progress counters read early and checked.
4. Full generation compared with the reference answers.
