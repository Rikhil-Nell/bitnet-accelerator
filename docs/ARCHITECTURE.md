# Architecture: BitNet b1.58 2B-4T on the Zybo Z7-10

Goal: **≥ 2 tokens/s decode**, short context (≤ 512 tokens), everything on the board.

## The one equation

Decoding one token is a matrix-vector product against every weight in the model,
and each weight is used exactly once. So the board is **memory-bandwidth bound**:

```
tokens/s ≈ sustained DDR read bandwidth / bytes of weights per token
```

Compute is cheap: a ternary weight is an add, a subtract or nothing.

| Per token | Params | Bytes |
|---|---|---|
| Ternary BitLinear (7 matrices × 30 layers) | 2.084 B | 521 MB at 2 bit/trit (417 MB at 1.6) |
| LM head (tied embedding) | 328 M | 328 MB at int8 (657 MB at F16 in the GGUF: does not fit) |
| **Total streamed** | | **849 MB** (745 MB with 1.6-bit packing) |
| KV cache (5 KV heads × 128, fp16) | | 77 KB per token of context, *read* per step |

Zybo Z7-10 DDR3L: 32-bit @ 1066 MT/s = 4.26 GB/s peak. At a realistic
2.5–3.4 GB/s that is **3–4 tok/s ceiling** before PS overheads, so 2 tok/s has margin.

## Block diagram

```
          ┌──────────────── PS (Cortex-A9 ×2, bare-metal) ────────────────┐
          │ embedding lookup, RMSNorm, act quant, RoPE, attention, softmax,│
          │ ReLU², residuals, argmax, UART token I/O, SD weight loader     │
          └──────┬──────────────────────────┬──────────────────────────────┘
          M_AXI_GP0 (CSRs)          S_AXI_HP0..HP3 (DDR, 64-bit each)
                 │                ┌─────────┼─────────┬─────────┐
          ┌──────▼──────┐   ┌─────▼───┐ ┌───▼─────┐ ┌─▼───────┐ ┌▼────────┐
          │ gemv_csr    │──▶│ unit 0  │ │ unit 1  │ │ unit 2  │ │ unit 3  │
          │ (AXI-Lite)  │   └─────────┘ └─────────┘ └─────────┘ └─────────┘
          └─────────────┘   each unit: AXI3 read master → FIFO → act buffer
                            + 32-lane ternary / 8-lane int8 datapath
                            → int32 out FIFO → AXI3 write master
```

* **Four identical units**, one per HP port. A matrix is split into four equal
  row ranges; unit *u* computes rows `[u·R, (u+1)·R)`. Units never share data,
  so there is no arbitration and no cross-unit wiring.
* Per unit, per command: (1) read the int8 activation vector `x` (K bytes) into
  its own activation buffer, (2) stream its rows of weights and accumulate one
  row at a time, (3) write one int32 per row to `y`.
* **4 ports × 64 bit × 150 MHz = 4.8 GB/s** of port bandwidth, which is more than
  the DDR can deliver, so the DDR is the only bottleneck.
* Fused calls: weights for `q|k|v` are stored as one 3840×2560 matrix and
  `gate|up` as one 13824×2560 matrix, so a layer is **4 commands**
  (qkv, o, gate_up, down) and a token is 120 commands + LM head.

## Weight formats in DDR

**Ternary (mode 0).** Row-major, 2 bits per trit, 32 trits per 64-bit word.
Trit `j` of a row is in bits `[2j+1:2j]` of word `j/32` of that row
(little-endian), coded as a signed 2-bit value: `00` = 0, `01` = +1, `11` = −1.
K must be a multiple of 32 (2560 and 6912 are).

**Int8 (mode 1, LM head).** Row-major int8, 8 weights per 64-bit word, plus one
float32 scale per row (applied on the PS).

Each unit's slice is 4 KiB aligned. Every slice size, `K`, and `R·4` used by
this model is a multiple of 128 B (one 16-beat AXI3 burst), so bursts never
cross a 4 KiB boundary.

## CSRs (AXI-Lite, base `0x43C0_0000`)

| Offset | Name | Meaning |
|---|---|---|
| 0x00 | CTRL | bit0 start (self-clearing) |
| 0x04 | STATUS | bit0 done (sticky until next start), bit1 busy, bits[7:4] per-unit done |
| 0x08 | MODE | 0 = ternary, 1 = int8 |
| 0x0C | K | input length in elements |
| 0x10 | ROWS | rows per unit (R) |
| 0x14 | X_ADDR | activation vector (int8, K bytes, 128 B aligned) |
| 0x18 | CYCLES | PL clock cycles of the last command (perf counter) |
| 0x1C | VERSION | engine ID / version |
| 0x20 + 8u | W_ADDR[u] | weight slice for unit u |
| 0x24 + 8u | Y_ADDR[u] | output slice for unit u (int32 × R) |

## PS software

Bare-metal C, one core first. Per BitLinear: quantize the activation (absmax
int8), write it to `x`, flush that cache line range, write CSRs, start, poll
done, invalidate `y`, dequantize with the per-tensor weight scale.
Everything else (norms, RoPE, attention over the KV cache, ReLU², residual adds,
argmax) runs on the A9 in float32.

Weights come from the **SD card** (FAT32, one packed file) and load into DDR at
boot with xilffs, not over JTAG. The PC side becomes a chat client: it tokenizes,
sends token ids over UART and prints the ids that come back.

## Correctness ladder

1. `tools/bitnet/engine.py` (NumPy) is the golden model. It already reproduces
   bitnet.cpp's answer ("The capital of France is Paris.").
2. RTL simulation against Python-generated vectors (random and real weights).
3. On board: random matrices checked against the A9 computing the same product.
4. On board: per-layer hidden states compared against `engine.py` dumps.
5. Full generation compared token-for-token against the emulated run.

## Later headroom

* 1.6-bit packing (5 trits/byte, decoded by a 256-entry LUT in the PL): −20 % bytes.
* Second A9 core for attention and norms while the PL streams the next matrix.
* Linux on the PS (UIO driver + reserved memory) once the bare-metal path works.
