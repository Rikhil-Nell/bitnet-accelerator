# Evaluation plan (agent + CI friendly)

Each level gates the next. Commands assume repo root.

## L0 — Golden reference (no Verilog)

| ID | Check | Command | Pass criteria |
|----|--------|---------|----------------|
| G0 | System 8×8 vector | `python tools/golden/ternary_gemv.py` | Prints `PASS` |
| G1 | Pack/unpack roundtrip | `python tools/eval/run_eval.py` | All unit checks pass |

**Owner:** Python golden is **source of truth** for linear math.

## L1 — RTL unit sim (Icarus)

| ID | Module | Command | Pass criteria |
|----|--------|---------|----------------|
| S0 | `ternary_decoder` | `make sim-decoder` | TB reports pass |
| S1 | `ternary_mac` | `make sim-mac` | TB reports pass |
| S2 | Full accelerator | `make sim-accelerator` | `TEST PASSED` in log |

Requires `iverilog` in PATH.

## L2 — RTL vs golden (future)

Parse `$display` from S2 or use VCD dump; `run_eval.py --compare-sim build/sim/bitnet_accelerator.log` (stub until log parser lands).

## L3 — Synthesis sanity (Vivado)

| ID | Check | Command | Pass criteria |
|----|--------|---------|----------------|
| Y0 | Elaborate RTL | `make synth-rtl` | No critical synth errors |
| Y1 | Utilization | report | LUT/BRAM within Z010 budget |

## L4 — Zynq integration

| ID | Check | Pass criteria |
|----|--------|----------------|
| Z0 | BD validates | PS7 + AXI-Lite + IP, address map unique |
| Z1 | Bitstream builds | `make bitstream` |
| Z2 | Program JTAG | `make program` |
| Z3 | PS bare-metal | UART banner + one GEMV vs golden |

## L5 — Host-in-the-loop (capstone demo)

| ID | Check | Pass criteria |
|----|--------|----------------|
| H0 | PC sends activations | UART protocol checksum OK |
| H1 | PL result | Matches `gemv()` within 0 |
| H2 | (Stretch) SD weight slice | One layer from file |

## Regression rule for agents

After any RTL change under `rtl/des/`:

1. `python tools/golden/ternary_gemv.py`
2. `make sim-all` (when Icarus available)
3. If BD changed: `make bitstream` before claiming hardware-ready
