# Board results v1 (2026-11-01)

BitNet b1.58 2B-4T running entirely on the Zybo Z7-10. Every ternary BitLinear and
the int8 LM head run on the 4-unit PL GEMV engine; norms, RoPE, attention, ReLU² and
argmax run on Cortex-A9 #0. The PC only tokenizes (`tools/bitnet/chat.py`).

## Answers (greedy, `make chat`)

| Prompt | Board answer | Prompt | Decode |
|---|---|---|---|
| What is the capital of France? | The capital of France is Paris. | 14 tok, 2.8 s | 3.41 tok/s |
| How many legs does a spider have? | Spiders have eight legs. This characteristic is common to most species of spiders. | 15 tok, 3.0 s | 3.33 tok/s |
| Who wrote Romeo and Juliet? | Romeo and Juliet was written by William Shakespeare. It is one of his most famous plays… | 13 tok, 2.6 s | 3.12 tok/s (48 tok) |
| What is 12 times 7? | 12 times 7 equals 84. | 15 tok, 3.0 s | 3.39 tok/s |
| Name a planet in our solar system that has rings. | Saturn is the planet in our solar system that has rings. | 18 tok, 3.6 s | 3.30 tok/s |
| What gas do plants absorb from the air? | Plants absorb carbon dioxide (CO2) from the air through a process called photosynthesis… | 16 tok, 3.2 s | 3.08 tok/s (48 tok) |
| Explain in two sentences why the sky is blue. | The sky appears blue because molecules in the Earth's atmosphere scatter sunlight… | 17 tok, 3.4 s | 3.16 tok/s |

Decode speed falls slowly with context because attention runs on the A9.
Wording can diverge from the NumPy reference after 10–20 tokens at near-ties
(float summation order); every engine matmul is bit-exact (`gemv_bench.txt`).

## Engine (gemv_bench, PL clock 142.857 MHz)

| Per token | Time | Bandwidth |
|---|---|---|
| 30 × qkv | 18.4 ms | ~4.06 GB/s |
| 30 × o | 12.8 ms | ~3.91 GB/s |
| 30 × gate_up | 68.4 ms | ~3.92 GB/s |
| 30 × down | 32.9 ms | ~4.07 GB/s |
| int8 LM head (row split, see below) | 81.5 ms | ~4.0 GB/s |
| **Total matrix work** | **219 ms** | ceiling ≈ 4.6 tok/s |

DDR3L peak is 4.27 GB/s (32-bit, 1066 MT/s), so the engine runs at ~95 % of it.
Unit slices that start a multiple of 32 KB apart collide in one DDR bank and drop to
2.25 GB/s; the LM head is therefore issued as 4 × 32016 + 4 × 48 rows.

Resources: 28 % LUT, 33 % BRAM, 45 % DSP; timing met (+0.43 ns).

## Baseline for comparison (tag `baseline-8x8-uart`)

Prefill 5844 s for 14 tokens, 519 s per decoded token, weights loaded over JTAG in
18 min. Now: prefill ~2.8 s, ~0.30 s per token, SD load 41 s.
