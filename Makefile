# BitNet on Zybo Z7-10. Run `make help`.
PYTHON ?= python
COM    ?= COM4
PS     := powershell -NoProfile
ENV    := . '$(CURDIR)/tools/setup/env.ps1';

.PHONY: help board chat web demo demo-replay bench sim bitstream ps-app ps-app-rebuild check-ftdi list-hw program run-ps srst pack generate-emulated bitnet-cpp-build

help:
	@echo "Run it (board on, SD card with BITNET.BIN in the slot)"
	@echo "  board [FCLK_DIV=7]  program the FPGA and boot the model (engine at 100 MHz; 7 = 142.9 MHz, 5 V supply)"
	@echo "  chat [COM=COM4]     chat with the model from this PC"
	@echo "  demo [COM=COM4]     web page: write-up + live chat/profile, http://127.0.0.1:8000"
	@echo "  demo-replay         the same page without a board (recorded answer)"
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
	@echo "  srst                JTAG system reset of the PS (A9 debug port stuck; then make board)"
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

# Program, then (after stopping Vivado's hw_server, which can leave the A9 debug port
# unreachable for xsdb) download and start the firmware.
board: export BITNET_ELF := $(call WS_ELF,bitnet_zybo)
board: program
	$(PS) -File tools/setup/stop_hw_server.ps1
	$(PS) -File tools/setup/run_ps_xsdb.ps1
	$(PYTHON) tools/bitnet/uart_log.py --port $(COM) --until "READY|HALT" --timeout 300
	$(if $(FCLK_DIV),$(PYTHON) tools/bitnet/board_cmd.py "F $(FCLK_DIV)" --until "^OK" --timeout 5)

chat:
	$(PYTHON) tools/bitnet/chat.py --port $(COM)

web:
	cd web/ui && npm install && npm run build

demo: web
	$(PYTHON) web/server.py --port $(COM)

demo-replay: web
	$(PYTHON) web/server.py --replay

bench: export BITNET_ELF := $(call WS_ELF,gemv_bench)
bench: program
	$(PS) -File tools/setup/stop_hw_server.ps1
	$(PS) -File tools/setup/run_ps_xsdb.ps1
	$(PYTHON) tools/bitnet/uart_log.py --port $(COM) --until "BENCH (PASS|FAIL)" --timeout 600

srst: check-ftdi
	$(PS) -File tools/setup/stop_hw_server.ps1
	$(PS) -File tools/setup/run_ps_xsdb.ps1 -Script fpga/scripts/srst_zybo.tcl

check-ftdi:
	$(PS) -File tools/setup/check_ftdi_drivers.ps1

list-hw: check-ftdi
	$(PS) -Command "$(ENV) vivado -mode batch -nojournal -source fpga/scripts/list_hw_targets.tcl -log build/vivado/list_hw.log"

program: check-ftdi
	$(PS) -Command "$(ENV) vivado -mode batch -nojournal -source fpga/scripts/program_zybo.tcl -log build/vivado/program.log"

run-ps: export BITNET_ELF := $(ELF)
run-ps: check-ftdi
	$(PS) -File tools/setup/stop_hw_server.ps1
	$(PS) -File tools/setup/run_ps_xsdb.ps1

pack:
	$(PYTHON) tools/bitnet/pack_board.py

generate-emulated:
	$(PYTHON) tools/bitnet/generate.py --backend emulated -p "$(if $(PROMPT),$(PROMPT),What is the capital of France?)"

bitnet-cpp-build:
	$(PS) -File tools/setup/build_bitnet_cpp.ps1
