# Zybo Z7-10 block design: PS7 (GP0 + HP0..HP3 @ 64 bit) + gemv_top -> bitstream + XSA.
#   vivado -mode batch -source fpga/scripts/build_bd.tcl
# Env: BITNET_FCLK_MHZ (default 150).
set repo_root  [file normalize [file join [info script] .. .. ..]]
set rtl_dir    [file join $repo_root rtl]
set proj_dir   [file join $repo_root build bitnet_zybo]
set out_dir    [file join $repo_root build vivado]
set board_repo "D:/xilinx-tools/vivado-boards/new/board_files"
set fclk_mhz   150
if {[info exists ::env(BITNET_FCLK_MHZ)] && $::env(BITNET_FCLK_MHZ) ne ""} {
  set fclk_mhz $::env(BITNET_FCLK_MHZ)
}
set NU 4
file mkdir $out_dir

set_param board.repoPaths [list $board_repo]
create_project bitnet_zybo $proj_dir -part xc7z010clg400-1 -force
set_property board_part digilentinc.com:zybo-z7-10:part0:1.2 [current_project]
add_files -norecurse [list \
  [file join $rtl_dir axi3_rd.v] \
  [file join $rtl_dir axi3_wr.v] \
  [file join $rtl_dir gemv_unit.v] \
  [file join $rtl_dir gemv_top.v]]
update_compile_order -fileset sources_1

create_bd_design "design_1"

# ---------------- PS7 ----------------
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} $ps
set cfg [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $fclk_mhz]
for {set u 0} {$u < $NU} {incr u} {
  lappend cfg CONFIG.PCW_USE_S_AXI_HP$u {1} CONFIG.PCW_S_AXI_HP${u}_DATA_WIDTH {64}
}
set_property -dict $cfg $ps

# ---------------- engine + glue ----------------
set eng [create_bd_cell -type module -reference gemv_top gemv]
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_fclk0]
set ic  [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 csr_ic]
set_property CONFIG.NUM_MI {1} $ic

set clk [get_bd_pins ps7/FCLK_CLK0]
connect_bd_net $clk [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins gemv/aclk] \
  [get_bd_pins rst_fclk0/slowest_sync_clk] \
  [get_bd_pins csr_ic/ACLK] [get_bd_pins csr_ic/S00_ACLK] [get_bd_pins csr_ic/M00_ACLK]
for {set u 0} {$u < $NU} {incr u} {
  connect_bd_net $clk [get_bd_pins ps7/S_AXI_HP${u}_ACLK]
}
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins rst_fclk0/ext_reset_in]
connect_bd_net [get_bd_pins rst_fclk0/peripheral_aresetn] [get_bd_pins gemv/aresetn] \
  [get_bd_pins csr_ic/S00_ARESETN] [get_bd_pins csr_ic/M00_ARESETN]
connect_bd_net [get_bd_pins rst_fclk0/interconnect_aresetn] [get_bd_pins csr_ic/ARESETN]

connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins csr_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins csr_ic/M00_AXI] [get_bd_intf_pins gemv/S_AXI]
for {set u 0} {$u < $NU} {incr u} {
  connect_bd_intf_net [get_bd_intf_pins gemv/M${u}_AXI] [get_bd_intf_pins ps7/S_AXI_HP$u]
}

# ---------------- addresses ----------------
assign_bd_address
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data] -filter {NAME =~ "*gemv*"}]
set_property offset 0x43C00000 $seg
set_property range 4K $seg
puts "CSR segment: [get_property offset $seg] range [get_property range $seg]"
foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces gemv/M*_AXI]] {
  puts "HP segment: $s offset [get_property offset $s] range [get_property range $s]"
}

validate_bd_design
save_bd_design

set bd_file [get_files design_1.bd]
make_wrapper -files $bd_file -top
add_files -norecurse [glob [file join $proj_dir bitnet_zybo.gen sources_1 bd design_1 hdl design_1_wrapper.v]]
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ---------------- build ----------------
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "Synthesis failed" }
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "Implementation failed" }

open_run impl_1
report_utilization -file [file join $out_dir utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $out_dir utilization_hier.rpt]
report_timing_summary -max_paths 10 -file [file join $out_dir timing.rpt]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "TIMING: FCLK0=${fclk_mhz} MHz WNS=$wns ns WHS=$whs ns"

file copy -force [file join $proj_dir bitnet_zybo.runs impl_1 design_1_wrapper.bit] [file join $out_dir bitnet_zybo.bit]
write_hw_platform -fixed -force -include_bit -file [file join $out_dir bitnet_zybo.xsa]
file copy -force [glob [file join $proj_dir bitnet_zybo.gen sources_1 bd design_1 ip design_1_ps7_0 ps7_init.tcl]] \
  [file join $out_dir ps7_init.tcl]
if {$wns < 0 || $whs < 0} {
  error "Timing not met (WNS=$wns WHS=$whs) at ${fclk_mhz} MHz"
}
puts "BUILD OK: [file join $out_dir bitnet_zybo.bit]"
