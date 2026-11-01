# Iteration log

## Step 1 — P1 AXI + Vivado synth (current)

**Goal:** Software can drive the accelerator like the testbench, via AXI-Lite; Vivado proves the top synthesizes on Z010.

| Task | Command / artifact |
|------|---------------------|
| L0 golden | `make eval` |
| AXI top RTL | `bitnet_axi_lite.v`, `bitnet_ip_top.v` |
| Vivado synth | `make vivado-synth` → `build/vivado/utilization.rpt` |
| RTL sim | `make sim-all` (xsim fallback) |

**Register map (byte offsets):**

| Offset | Write | Read |
|--------|-------|------|
| `0x00` | bit0 = START | — |
| `0x04` | — | STATUS `{busy, done}` in bits [1:0] |
| `0x10` | ACT: `[10:8]` addr, `[7:0]` data | — |
| `0x14` | WGT: `[18:16]` addr, `[15:0]` data | — |
| `0x18` | OUT index `[2:0]` | — |
| `0x1C` | — | OUT data (sign-extended 20-bit) |

## Step 2 — Block design + bitstream (done)

`make bitstream` → `build/vivado/bitnet_zybo.bit`, `bitnet_zybo.xsa`

## Host kernel (Step 1b)

`make test-kernel` — PC BitLinear matches golden + NumPy reference; RTL sim when iverilog installed.

## Step 3 — PS bare-metal (build done; board run pending JTAG)

- `make ps-app` → `D:/xilinx-tools/bitnet_vitis_ws_clean/system_8x8_test/build/system_8x8_test.elf`
- Program PL: `make program` (requires JTAG target visible to Vivado)
- Run ELF via Vitis **Run/Debug** (Python XSDB) or SD boot image; UART **115200** → `PASS: system_8x8 on PL accelerator`
