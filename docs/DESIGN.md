# BitNet Zybo accelerator — design decisions (living doc)

Target: **Digilent Zybo Z7-10** (`xc7z010clg400-1`), **1 GB DDR3**, capstone demo of **BitNet b1.58-style W2A8** on low-cost Zynq.

## North-star demo

Power on → (optional UART/HDMI) → run **one constrained inference step** (short prompt, few tokens) with **layer-wise weight streaming from microSD**, **PL accelerates BitLinear GEMV**, **PS + host reference** handle the rest. Optimized throughput is **not** a goal; **correctness + feasibility** are.

## Architecture split (locked)

| Layer | Where | Why |
|-------|--------|-----|
| BitLinear GEMV/GEMM (W2A8) | **PL** (`bitnet_accelerator` → AXI slave) | Project focus; energy/latency story |
| RoPE, softmax, RMSNorm, SwiGLU | **PS** (bare-metal first) | Too heavy to rewrite in PL v1 |
| Golden reference / BitNet slice | **Host PC** (Python) | Fast iteration; compares PL vs NumPy |
| Full model weights | **microSD** + DDR | 2B ternary ~400 MB; stream per layer |
| Boot / debug | **USB-JTAG + UART** | No SD required for bring-up |

## Ternary weight encoding (repo convention)

**RTL / testbench / golden model** (do not change without updating all three):

| 2-bit code | Value |
|------------|--------|
| `00` | 0 |
| `01` | +1 |
| `10` | −1 |
| `11` | 0 (reserved) |

Packing: weight `i` occupies bits `[2i+1 : 2i]` in a 16-bit word (8 weights per word).

**Microsoft bitnet.cpp I2_S** uses a **different** map (`00→−1`, `01→0`, `10→+1`, interleaved blocks). Importing real checkpoints requires an explicit **`i2s_to_capstone()`** converter — not raw memcpy.

## Accelerator micro-architecture (phased)

| Phase | PL capability | PS capability |
|-------|----------------|---------------|
| **P0** (now) | Fixed **8×8** INT8×ternary MAC, FSM, register/file ports | Load memories via AXI-Lite, poll DONE |
| **P1** | AXI4-Lite wrapper + IP in Zynq BD | Bare-metal driver, UART echo |
| **P2** | Parameterized **K**, single output lane time-multiplexed | Layer invoke from host |
| **P3** | Optional DMA (HP port) for DDR buffers | Streaming loader from SD |

**Z010 constraint:** ~17.6k LUTs — stay **single-lane**, on-chip BRAM weights for demo sizes; avoid wide systolic arrays.

## AXI4-Lite register map (planned)

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| `0x00` | `CTRL` | W | Write `1` → START pulse (auto-clear) |
| `0x04` | `STATUS` | R | bit0 DONE, bit1 BUSY, bit2 ERROR |
| `0x08` | `CFG` | R/W | Reserved (future K, N) |
| `0x0C` | `IRQ` | R/W | Optional interrupt enable |
| `0x10` | `ACT_BASE` | R/W | PS DDR pointer (future DMA) |
| `0x14` | `WGT_BASE` | R/W | PS DDR pointer (future DMA) |
| `0x18` | `OUT_BASE` | R/W | PS DDR pointer (future DMA) |

Memory-mapped **on-chip** activation/weight windows may remain as today until DMA lands.

## Clocking / constraints

- **Block designs:** constrain **PS `FCLK_CLK0`** (typical 50–125 MHz on Zybo presets), not only the 125 MHz **K17** PL crystal ([Digilent master XDC](https://github.com/Digilent/digilent-xdc/blob/master/Zybo-Z7-Master.xdc) on `D:\xilinx-tools\digilent-xdc`).
- Refactor **async memory reads** → **sync BRAM** before pushing Fmax.

## Tooling (2026.1)

- Install: `D:\Xilinx\2026.1\`
- **No `xsct`** — Vitis **Python API** + **Python XSDB** for run/debug.
- Agent loop: **`make golden`** → **`make sim-*`** → **`make bitstream`** (when `fpga/` exists) → **`make program`** (later).

## Verification strategy

See `tools/eval/EVAL_PLAN.md`. Golden Python is source of truth for linear ops; Verilog TB must match `tools/golden/vectors/system_8x8.json`.

## Explicit non-goals (v1)

- Full 2B chat at human token rate on PL alone
- bitnet.cpp port without conversion layer
- HDMI GUI before UART bring-up works
- HLS / Vitis AI
