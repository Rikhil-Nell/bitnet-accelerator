# BitNet on Zybo Z7-10. Run `make help`.
PYTHON ?= python
COM    ?= COM4
PS     := powershell -NoProfile
ENV    := . '$(CURDIR)/tools/setup/env.ps1';

.PHONY: help board chat bench sim bitstream ps-app ps-app-rebuild check-ftdi list-hw program run-ps generate-emulated bitnet-cpp-build

help:
	@echo "Run it (board on, SD card with BITNET.BIN in the slot)"
	@echo "  board               program the FPGA and boot the model (~45 s to READY)"
	@echo "  chat [COM=COM4]     chat with the model from this PC"
	@echo "  bench               engine benchmark on the board (prints to COM4)"
	@echo "Engine"
	@echo "  sim [STALL=25]      xsim gemv_top vs NumPy vectors (random AXI stalls)"
	@echo "  bitstream [FCLK=150] block design + synth + impl + XSA (build/vivado/)"
	@echo "  ps-app [APP=gemv_bench]        Vitis platform (from XSA) + bare-metal app"
	@echo "  ps-app-rebuild [APP=...]       rebuild app sources only"
	@echo "Board / JTAG"
	@echo "  check-ftdi          FTDI bus/port driver versions match (guards all JTAG targets)"
	@echo "  list-hw             list JTAG targets"
	@echo "  program             program the bitstream (build/vivado/bitnet_zybo.bit)"
	@echo "  run-ps ELF=...      ps7_init + download and run an ELF on A9 #0"
	@echo "Model"
	@echo "  pack                write D:/xilinx-tools/BITNET.BIN from the GGUF (copy to SD root)"
	@echo "  generate-emulated PROMPT=...   reference generation on the PC (NumPy)"
	@echo "  bitnet-cpp-build    build Microsoft bitnet.cpp (reference)"

sim:
	$(PS) -File fpga/scripts/sim_gemv.ps1 -Stall $(if $(STALL),$(STALL),25)

bitstream: export BITNET_FCLK_MHZ := $(FCLK)
bitstream:
	$(PS) -Command "$(ENV) vivado -mode batch -nojournal -source fpga/scripts/build_bd.tcl -log build/vivado/bitstream.log"

APP ?= gemv_bench
VITIS = $(PS) -File tools/setup/vitis_run.ps1 -App $(APP) -Script

ps-app:
	$(VITIS) fpga/scripts/build_ps_platform.py

ps-app-rebuild:
	$(VITIS) fpga/scripts/rebuild_ps_app_only.py

WS_ELF = D:/xilinx-tools/vitis_ws/$(1)/build/$(1).elf

board: program
	$(MAKE) run-ps ELF=$(call WS_ELF,bitnet_zybo)
	$(PYTHON) tools/bitnet/uart_log.py --port $(COM) --until "READY|HALT" --timeout 300

chat:
	$(PYTHON) tools/bitnet/chat.py --port $(COM)

bench: program
	$(MAKE) run-ps ELF=$(call WS_ELF,gemv_bench)
	$(PYTHON) tools/bitnet/uart_log.py --port $(COM) --until "BENCH (PASS|FAIL)" --timeout 600

check-ftdi:
	$(PS) -File tools/setup/check_ftdi_drivers.ps1

list-hw: check-ftdi
	$(PS) -Command "$(ENV) vivado -mode batch -nojournal -source fpga/scripts/list_hw_targets.tcl -log build/vivado/list_hw.log"

program: check-ftdi
	$(PS) -Command "$(ENV) vivado -mode batch -nojournal -source fpga/scripts/program_zybo.tcl -log build/vivado/program.log"

run-ps: export BITNET_ELF := $(ELF)
run-ps: check-ftdi
	$(PS) -File tools/setup/run_ps_xsdb.ps1

pack:
	$(PYTHON) tools/bitnet/pack_board.py

generate-emulated:
	$(PYTHON) tools/bitnet/generate.py --backend emulated -p "$(if $(PROMPT),$(PROMPT),What is the capital of France?)"

bitnet-cpp-build:
	$(PS) -File tools/setup/build_bitnet_cpp.ps1
