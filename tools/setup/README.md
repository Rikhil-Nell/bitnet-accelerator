# Toolchain setup (Zybo Z7-10, install on `D:`)

## Already on this machine

| Item | Location |
|------|----------|
| Digilent board files (Zybo Z7-10) | `D:\xilinx-tools\vivado-boards\new\board_files\zybo-z7-10\A.0` |
| Ross skills repo | `D:\xilinx-tools\ross-ai-assistant` |
| Batch install config (draft) | `tools/setup/install_config_zybo.txt` |
| Environment helper | `tools/setup/env.ps1` |

## Disk budget

- **D: ~154 GB free** — enough for a **Zynq-7000–only** Vivado + Vitis embedded install if you do **not** install every device family.
- Full unified SFD images are **100+ GB**; use the **web installer** and select **Zynq-7000** only.

## Step 1 — AMD account + web installer

1. Create/login: [AMD adaptive downloads](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools/2025-1.html)
2. Download **Windows Self Extracting Web Installer** (2025.1 or 2025.1.1).
3. Save to `D:\xilinx-tools\downloads\`
4. Run the `.exe` (GUI is fine for first time):
   - **Download and Install**
   - Edition: **Vitis Unified** (includes Vivado + embedded Vitis)
   - Install location: **`D:\Xilinx`** (no spaces)
   - Devices: **Zynq-7000** only
   - Skip Versal / UltraScale+ / etc.

Or extract the web client and batch-install (Admin PowerShell):

```powershell
cd D:\xilinx-tools\downloads
.\FPGAs_AdaptiveSoCs_Unified_Web_*.exe -y -gm2 -nr -InstallPath="D:\xilinx-tools\wi-client"
cd D:\xilinx-tools\wi-client\bin
.\xsetup.bat --agree XilinxEULA,3rdPartyEULA --batch ConfigGen --edition "Vitis Unified Software Platform" --location D:\Xilinx
# Edit the generated install_config.txt, then:
.\xsetup.bat --agree XilinxEULA,3rdPartyEULA --batch Install --config <path-to-config>
```

**Do not use `xsct`** on 2026.x; use **Vitis Python API** + **Python XSDB** (see UG1400).

## Step 2 — Board files

Point Vivado at the repo (no copy needed):

- **Settings → Vivado Store → Board Repository →** add `D:\xilinx-tools\vivado-boards\new\board_files`

Or copy `zybo-z7-10` into `D:\Xilinx\<version>\Vivado\data\boards\board_files\`.

## Step 3 — Verify install

```powershell
. C:\Users\Rikhil Nellimarla\Projects\bitnet-accelerator\tools\setup\env.ps1
vivado -version
```

## Step 4 — Ross (optional, after Vivado works)

1. [Ross downloads](https://www.amd.com/en/support/downloads/ross-agentic-ai.html) — install **Vivado AI Extension (.vsix)** in Cursor **or** standalone **vivado-mcp-server** exe.
2. Copy/link skills from `D:\xilinx-tools\ross-ai-assistant` per AMD README.

## Step 5 — Agent sim loop (no Vivado)

From repo root:

```powershell
make sim-decoder   # requires Icarus Verilog (`iverilog` in PATH)
```

Install Icarus (Admin): `choco install iverilog -y`  
Or Scoop: `scoop install iverilog`

## Bitstream note

Current RTL under `bitnet-fpga-accelerator/rtl/des/` is **not a complete FPGA top** (no Zynq PS7 block design, no pin constraints). Bitstream generation requires `fpga/` integration (next implementation step).
