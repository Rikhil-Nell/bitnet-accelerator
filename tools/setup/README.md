# tools/setup

Installation steps for a new machine are in the top-level [`SETUP.md`](../../SETUP.md).
These scripts are called by the Makefile; paths come from `local.mk` / environment variables.

| Script | Used by | What it does |
|---|---|---|
| `env.ps1` | most targets | loads the Vivado/Vitis environment (`XILINX_ROOT`, `XILINX_VERSION`) and the board-file path |
| `vitis_run.ps1` | `make ps-app*` | runs a Vitis Python script; stops leftover Vitis servers before and after |
| `run_ps_xsdb.ps1` | `make board`, `run-ps`, `srst` | XSDB with a minimal PATH; finds the second core's entry point in the ELF |
| `stop_hw_server.ps1` | `make board` | stops a lingering `hw_server` (it can leave the A9 debug port unreachable) |
| `check_ftdi_drivers.ps1` | every JTAG target | refuses to run when the FTDI bus and port driver versions differ |
| `link-board-files.ps1` | once, optional | links the Digilent board files into the Vivado install |
| `install_config_zybo.txt` | once, optional | batch-install config for a Zynq-7000-only Vivado/Vitis |
| `diagnose_zybo_usb.ps1`, `find_recent_bugcheck.ps1` | troubleshooting | USB/JTAG and crash diagnostics (`docs/ZYBO_JTAG.md`) |
