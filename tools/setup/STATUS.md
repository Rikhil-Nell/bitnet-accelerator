# Toolchain setup status (agent run)

Last updated: 2026-10-31 (prep pass)

## Completed

- [x] **Design doc** `docs/DESIGN.md` (architecture, encoding, register map, phases)
- [x] **Golden model + L0 evals** `tools/golden/`, `tools/eval/` — `python tools/eval/run_eval.py` **PASS**
- [x] **Makefile** targets: `make golden`, `make eval`, `make sim-accelerator`
- [x] **FPGA scaffold** `fpga/` + placeholder `build_bitstream.tcl`
- [x] **Digilent master XDC** cloned to `D:\xilinx-tools\digilent-xdc`
- [x] **UART protocol draft** `tools/host/UART_PROTOCOL.md`
- [x] **env.ps1** default version → **2026.1**
- [x] **Digilent board files** cloned to `D:\xilinx-tools\vivado-boards` (includes `zybo-z7-10/A.0`)
- [x] **Ross skills repo** cloned to `D:\xilinx-tools\ross-ai-assistant`
- [x] **Cursor rule** `.cursor/rules/fpga-zynq-bitnet.mdc`
- [x] **Makefile** smoke sim targets (`make sim-decoder`, `make sim-mac`)
- [x] **Install helpers** in `tools/setup/` (`env.ps1`, `link-board-files.ps1`, `install_config_zybo.txt`, README)
- [x] **Agent research** on Ross + script-first Vivado flows (subagent summary in chat)

## Blocked / needs you

- [ ] **Vivado/Vitis not installed** — no `D:\Xilinx` yet. AMD web installer requires **your AMD account login** in the browser; the agent cannot complete this headlessly.
  - Run: `tools/setup/start-amd-install.ps1` and save installer to `D:\xilinx-tools\downloads\`
  - Install to **`D:\Xilinx`**, devices **Zynq-7000 only**
- [ ] **Icarus Verilog** — not in PATH. Scoop install hit **Windows Defender** on the upstream installer; Chocolatey needs **Admin** + lock cleared. Run **Admin PowerShell**: `choco install iverilog -y`
- [ ] **Ross VSIX / MCP** — download from AMD Ross page and install in Cursor after Vivado works

## Bitstream from *current* RTL only?

**No — not yet.** The repo has synthesizable **blocks** (`bitnet_accelerator`, memories, FSM) but **no**:

- Zynq **PS7 block design** + AXI hookup
- **Top-level** wrapper and **XDC** constraints for Zybo
- Vivado **batch Tcl** / `fpga/` project

The agent can generate a bitstream **independently** only **after** that integration landings (next engineering step). Synthesis of `bitnet_accelerator` alone is not a valid Zynq bitstream.

## Disk

- C: ~42 GB free, D: ~154 GB free — use **D:** for Xilinx installs and downloads.
