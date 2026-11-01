# Fast RTL sim (Icarus Verilog). Vivado bitstream targets come with fpga/ integration.
IVERILOG ?= iverilog
VVP      ?= vvp
RTL      := bitnet-fpga-accelerator/rtl/des
TB       := bitnet-fpga-accelerator/rtl/tb
BUILD    := build/sim

.PHONY: sim-decoder sim-mac sim-accelerator sim-all golden eval check-iverilog list-hw uart-listen run-ps board-bringup

PYTHON ?= python
COM    ?= COM3

check-iverilog:
	@$(IVERILOG) -V >/dev/null 2>&1 || (echo "Install Icarus: choco install iverilog -y (Admin) or scoop install iverilog" && exit 1)

$(BUILD):
	@mkdir -p $(BUILD)

sim-decoder: check-iverilog | $(BUILD)
	$(IVERILOG) -g2012 -o $(BUILD)/ternary_decoder.vvp $(RTL)/ternary_decoder.v $(TB)/ternary_decoder_tb.v
	$(VVP) $(BUILD)/ternary_decoder.vvp

sim-mac: check-iverilog | $(BUILD)
	$(IVERILOG) -g2012 -o $(BUILD)/ternary_mac.vvp $(RTL)/ternary_mac.v $(TB)/ternary_mac_tb.v
	$(VVP) $(BUILD)/ternary_mac.vvp

sim-accelerator: check-iverilog | $(BUILD)
	$(IVERILOG) -g2012 -o $(BUILD)/bitnet_accelerator.vvp \
		$(RTL)/ternary_decoder.v $(RTL)/ternary_mac.v $(RTL)/accumulator.v \
		$(RTL)/activation_memory.v $(RTL)/weight_memory.v $(RTL)/output_memory.v \
		$(RTL)/controller_fsm.v $(RTL)/bitnet_accelerator.v \
		$(TB)/bitnet_accelerator_tb.v
	$(VVP) $(BUILD)/bitnet_accelerator.vvp

sim-all:
	$(PYTHON) tools/eval/run_sim_all.py

sim-xsim:
	powershell -NoProfile -Command ". '$(CURDIR)/tools/setup/env.ps1'; vivado -mode batch -source fpga/scripts/sim_rtl_xsim.tcl -log build/sim/xsim.log -journal build/sim/xsim.jou"

golden:
	$(PYTHON) tools/golden/ternary_gemv.py

capstone-check:
	$(PYTHON) tools/eval/capstone_check.py

eval:
	$(PYTHON) tools/eval/run_eval.py
	$(PYTHON) tools/host/test_model_kernel.py
	$(PYTHON) tools/host/mmio_runner.py
	$(PYTHON) tools/host/test_i2s_convert.py

vivado-synth:
	powershell -NoProfile -Command ". '$(CURDIR)/tools/setup/env.ps1'; vivado -mode batch -source fpga/scripts/synth_rtl.tcl -log build/vivado/synth.log -journal build/vivado/synth.jou"

test-kernel:
	$(PYTHON) tools/host/test_model_kernel.py

test-mmio:
	$(PYTHON) tools/host/mmio_runner.py

test-i2s:
	$(PYTHON) tools/host/test_i2s_convert.py

test-bitnet-smoke:
	$(PYTHON) tools/host/bitnet_smoke.py

test-bitnet-layer:
	$(PYTHON) tools/host/test_bitnet_layer.py

bitstream:
	powershell -NoProfile -Command ". '$(CURDIR)/tools/setup/env.ps1'; vivado -mode batch -source fpga/scripts/build_zybo_bd.tcl -log build/vivado/bitstream.log -journal build/vivado/bitstream.jou"

export-xsa:
	powershell -NoProfile -Command ". '$(CURDIR)/tools/setup/env.ps1'; vivado -mode batch -source fpga/scripts/export_xsa.tcl -log build/vivado/export_xsa.log"

PS_APP ?= system_8x8_test

ps-app:
	cmd /c "call D:/Xilinx/2026.1/Vitis/settings64.bat >nul 2>&1 && set VITIS_EMBEDDED_INSTALL=true&& set BITNET_PS_APP=$(PS_APP)&& cd /d $(CURDIR) && vitis -s fpga/scripts/build_ps_platform.py"

ps-app-rebuild:
	cmd /c "call D:/Xilinx/2026.1/Vitis/settings64.bat >nul 2>&1 && set VITIS_EMBEDDED_INSTALL=true&& set BITNET_PS_APP=$(PS_APP)&& cd /d $(CURDIR) && vitis -s fpga/scripts/rebuild_ps_app_only.py"

stage-board:
	powershell -NoProfile -File tools/setup/stage_board_assets.ps1

check-ftdi:
	powershell -NoProfile -File tools/setup/check_ftdi_drivers.ps1

list-hw: check-ftdi
	powershell -NoProfile -Command ". '$(CURDIR)/tools/setup/env.ps1'; vivado -mode batch -source fpga/scripts/list_hw_targets.tcl -log build/vivado/list_hw.log"

check-jtag-log:
	powershell -NoProfile -File tools/setup/parse_list_hw_log.ps1

list-hw-safe: check-ftdi
	powershell -NoProfile -File tools/setup/stop_hw_server.ps1
	powershell -NoProfile -Command ". '$(CURDIR)/tools/setup/env.ps1'; vivado -mode batch -source fpga/scripts/list_hw_targets.tcl -log build/vivado/list_hw.log"

program: check-ftdi
	powershell -NoProfile -Command ". '$(CURDIR)/tools/setup/env.ps1'; vivado -mode batch -source fpga/scripts/program_zybo.tcl -log build/vivado/program.log -journal build/vivado/program.jou"

run-ps: check-ftdi
	powershell -NoProfile -File tools/setup/run_ps_xsdb.ps1

board-bringup:
	powershell -NoProfile -File tools/setup/board_bringup.ps1 -Com $(COM)

board-test:
	powershell -NoProfile -File tools/setup/board_bringup.ps1 -Com $(COM) -Listen

uart-listen:
	$(PYTHON) tools/host/uart_board_test.py --port $(COM)

# BitNet 2B inference on PL (see docs/BITNET_BOARD_INFERENCE.md)
pack-bitnet-weights:
	$(PYTHON) tools/bitnet/pack_weights.py

bitlinear-ps-app:
	$(MAKE) ps-app-rebuild PS_APP=bitlinear_server

load-bitnet-weights: check-ftdi
	powershell -NoProfile -File tools/setup/load_bitnet_weights.ps1 2>&1 | Tee-Object -FilePath build/bitnet/load_weights.log

run-bitlinear-server: check-ftdi
	powershell -NoProfile -File tools/setup/run_bitlinear_server.ps1

bitnet-generate-emulated:
	$(PYTHON) tools/bitnet/generate.py --backend emulated $(if $(PROMPT),-p "$(PROMPT)",)

bitnet-generate-board:
	$(PYTHON) tools/bitnet/generate.py --backend board --port $(COM) --fast $(if $(PROMPT),-p "$(PROMPT)",) -n $(if $(N),$(N),1) --log build/bitnet/board_generate.json

bitnet-board-check:
	$(PYTHON) tools/bitnet/board_check.py --port $(COM)

bitnet-test-tiling:
	$(PYTHON) tools/bitnet/test_tiling.py

bitnet-test-fused:
	$(PYTHON) tools/bitnet/test_fused_gemm.py

bitnet-cpp-build:
	powershell -NoProfile -File tools/setup/build_bitnet_cpp.ps1 2>&1 | Tee-Object -FilePath build/bitnet/setup_env.log
