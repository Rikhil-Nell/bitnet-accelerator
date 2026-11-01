# BitNet board inference — performance roadmap

## Where time goes today (8×8 PL, PS-driven MMIO)

| Bottleneck | Effect |
|------------|--------|
| **UART payloads** | Each matmul ships ~2.5–10 KiB of `int32` outputs back to the PC at ~921600 baud (~90 KiB/s practical). Dozens of matmuls per token → seconds even if PL were free. |
| **PS MMIO per 8×8 tile** | Firmware loads weights/activations and polls STATUS for every tile. A 2560×2560 layer is 102400 tile runs; AXI-Lite latency dominates. |
| **Small array** | One tile is only 64 MACs; fixed overhead per tile is huge. |

Emulated inference on the PC is ~30 tok/s because NumPy does full matrices at once. The board path is structurally different.

## Target ~0.5 s/token — what has to change

**0.5 s/token is not reachable** with the current “host orchestrates every matmul over UART” design. It needs most of the decode loop on the board and a **PL tile sequencer** that streams weights from DDR without PS writes per tile.

### Tier 1 — shipped in this repo (no bitstream change)

- **`G` with M>1**: one UART transaction for Q+K+V or batched outputs (`engine.py` + `board_backend.py`).
- **Fused FFN gate+up (firmware v3)**: when M matrices share the same `N` and `K`, one walk over K-tiles loads activations once and runs each weight tile (`gemm_fused_same` in `bitlinear_server/main.c`). Cuts PL work ~2× for those layers vs two separate full passes.
- **Tighter STATUS polling** in firmware (512 polls vs 100k).
- **`generate.py --fast`**: skip emulation verify, DDR checksum scan, and output checksum.

Rebuild and reload firmware after changes:

```text
make bitlinear-ps-app
make run-bitlinear-server
```

Optional: re-pack weights if you enabled RMS norm blobs in `pack_weights.py` (adds ~1–2 MiB at end of `bitnet_weights.bin`).

### Tier 2 — next firmware (no PL change)

- **Full forward on PS**: keep hidden state and KV cache in DDR; host sends token id or embedding once, board returns next token id (4 bytes). Removes multi-megabyte UART per token.
- **Batch prefill**: `T>1` on `G` for prompt processing.

### Tier 3 — PL + block design (required for ~0.5 s/token)

- **AXI master tile engine**: CSR programs `{W_base, x_ptr, y_ptr, N, K}`; PL reads weight tiles from DDR, runs the existing 8×8 FSM, writes `int32` partials to DDR; PS gets an interrupt per row or per layer.
- **Optional wider PE** (16×16, 32×32) if LUT/DSP budget allows — linear speedup in compute.
- **Raise FCLK** if timing closure allows (current WNS was positive at 50 MHz).

Rough order-of-magnitude: Tier 1 often helps **2–4×** on wall time vs naive board mode; Tier 3 is needed for **100×+** to approach sub-second tokens.

## Measuring

- Firmware reports **global timer ticks** on each `G` response (host prints `board_seconds` in `board_backend`).
- Use `generate.py --log run.json` and compare `step_seconds` with and without `--fast`.
