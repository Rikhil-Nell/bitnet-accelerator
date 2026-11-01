# Board results v3 (2026-11-01): engine v3, streamed pass, dual core, speculation

All runs on the Zybo powered from a laptop USB port. That supply browns the board out
above ~3.2 GB/s of DDR streaming (bench power sweep: 2.69 GB/s at 83 MHz and 3.22 GB/s at
100 MHz survive, 111 MHz resets the board), so **the engine ran at 100 MHz** here instead
of 142.857 MHz. The profiler separates engine streaming from serial A9 work, which gives
the full-clock projection below.

## Measured at 100 MHz (greedy, answers identical to the reference)

| Prompt | Configuration | Tokens / pass | ms / token | tok/s |
|---|---|---|---|---|
| What is the capital of France? | streamed pass, 1 core, no speculation | 1.00 | 166.5 | 6.0 |
| What is the capital of France? | + speculation | 1.40 | 126.6 | 7.9 |
| What is the capital of France? | + speculation + 2 cores | 1.40 | 123.3 | 8.1 |
| Who wrote Romeo and Juliet? (48 tokens) | + speculation + 2 cores | 1.23 | 147.0 | 6.8 |

Prompt reading (two tokens per pass): 14 tokens in 1.08 s (77 ms/token; v1: 200 ms/token).

Per pass (France, 2 cores): engine stream 153.9 ms (490 MB at 3.2 GB/s, with 52.7 ms of
A9 epilogue work hidden inside it), serial A9 work 18.6 ms (attention 7.6, quantize 6.8,
RoPE 1.6, norm stats 1.4, the rest < 1).

## Measured at 83 MHz (default boot clock on USB power), CPU1 takes position 2

Per-unit statistics merged in a fixed order make the output reproducible run to run.
Speculation is exact: with guessing switched off ('N') the same pass gives the same text.

| Prompt | Tokens / pass | ms / pass | stream (PL + hidden A9) | serial A9 | ms / token |
|---|---|---|---|---|---|
| France (8 tokens) | 1.40 | 200.7 | 184.4 | 16.3 | 143 |
| Romeo and Juliet (48) | 1.27 | 207.2 | 184.5 | 22.7 | 163 |
| Romeo and Juliet, no guessing | 1.00 | 200.6 | 183.3 | 17.3 | 201 |

## Projection to 142.857 MHz (5 V supply)

The stream becomes 490 MB / 4.05 GB/s = 121 ms (measured bandwidth at full clock, engine
v2 bench). Serial A9 work is clock-independent.

| Prompt | Pass | Tokens / pass | ms / token | tok/s |
|---|---|---|---|---|
| France | 121 + 16.3 ms | 1.40 | 98 | **10.2** |
| Romeo and Juliet (48 tokens) | 121 + 22.7 ms | 1.27 | 113 | **8.8** |

## Engine bench (v2 bitstream, full clock)

T5 / int8 shapes bit-exact against the A9; progress counters: rows reported done read
correctly while the engine streams; 4.05 GB/s with staggered slices (2.23 GB/s when
the four slices start a multiple of 32 KB apart).
