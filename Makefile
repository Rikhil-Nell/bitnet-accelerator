# BitNet on Zybo Z7-10. Run `make help`.
PYTHON ?= python
COM    ?= COM4
PS     := powershell -NoProfile
ENV    := . '$(CURDIR)/tools/setup/env.ps1';

.PHONY: help sim check-ftdi list-hw program run-ps generate-emulated bitnet-cpp-build

help:
	@echo "Engine"
	@echo "  sim [STALL=25]      xsim gemv_top vs NumPy vectors (random AXI stalls)"
	@echo "Board / JTAG"
	@echo "  check-ftdi          FTDI bus/port driver versions match (guards all JTAG targets)"
	@echo "  list-hw             list JTAG targets"
	@echo "  program             program the bitstream (build/vivado/bitnet_zybo.bit)"
	@echo "  run-ps ELF=...      ps7_init + download and run an ELF on A9 #0"
	@echo "Model"
	@echo "  generate-emulated PROMPT=...   reference generation on the PC (NumPy)"
	@echo "  bitnet-cpp-build    build Microsoft bitnet.cpp (reference)"

sim:
	$(PS) -File fpga/scripts/sim_gemv.ps1 -Stall $(if $(STALL),$(STALL),25)

check-ftdi:
	$(PS) -File tools/setup/check_ftdi_drivers.ps1

list-hw: check-ftdi
	$(PS) -Command "$(ENV) vivado -mode batch -nojournal -source fpga/scripts/list_hw_targets.tcl -log build/vivado/list_hw.log"

program: check-ftdi
	$(PS) -Command "$(ENV) vivado -mode batch -nojournal -source fpga/scripts/program_zybo.tcl -log build/vivado/program.log"

run-ps: check-ftdi
	$(PS) -Command "$$env:BITNET_ELF='$(ELF)'; & tools/setup/run_ps_xsdb.ps1"

generate-emulated:
	$(PYTHON) tools/bitnet/generate.py --backend emulated -p "$(if $(PROMPT),$(PROMPT),What is the capital of France?)"

bitnet-cpp-build:
	$(PS) -File tools/setup/build_bitnet_cpp.ps1
