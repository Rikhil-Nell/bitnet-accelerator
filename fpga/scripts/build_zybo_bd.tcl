# Zybo Z7-10: PS7 + AXI GP0 + protocol converter + bitnet_ip_top -> bitstream
set repo_root [file normalize [file join [info script] .. .. ..]]
set rtl_dir   [file join $repo_root bitnet-fpga-accelerator rtl des]
set proj_dir  [file join $repo_root build bitnet_zybo]
set out_dir   [file join $repo_root build vivado]
set board_repo "D:/xilinx-tools/vivado-boards/new/board_files"
file mkdir $proj_dir
file mkdir $out_dir

# Digilent Zybo Z7-10 board files (not in AMD install by default)
set_param board.repoPaths [list $board_repo]

set rtl_files [list \
  [file join $rtl_dir ternary_decoder.v] \
  [file join $rtl_dir ternary_mac.v] \
  [file join $rtl_dir accumulator.v] \
  [file join $rtl_dir activation_memory.v] \
  [file join $rtl_dir weight_memory.v] \
  [file join $rtl_dir output_memory.v] \
  [file join $rtl_dir controller_fsm.v] \
  [file join $rtl_dir control_registers.v] \
  [file join $rtl_dir bitnet_accelerator.v] \
  [file join $rtl_dir bitnet_axi_lite.v] \
  [file join $rtl_dir bitnet_ip_top.v] \
]

create_project bitnet_zybo $proj_dir -part xc7z010clg400-1 -force
set_property board_part digilentinc.com:zybo-z7-10:part0:1.2 [current_project]

add_files -norecurse $rtl_files
update_compile_order -fileset sources_1

create_bd_design "design_1"
set bd design_1

create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} \
  [get_bd_cells processing_system7_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_protocol_converter:2.1 axi_pc_0
set_property -dict [list \
  CONFIG.SI_PROTOCOL {AXI3} \
  CONFIG.MI_PROTOCOL {AXI4LITE} \
] [get_bd_cells axi_pc_0]

create_bd_cell -type module -reference bitnet_ip_top bitnet_ip_top_0

connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] \
  [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] \
  [get_bd_intf_pins axi_pc_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_pc_0/M_AXI] \
  [get_bd_intf_pins bitnet_ip_top_0/S_AXI]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
  [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
  [get_bd_pins axi_interconnect_0/ACLK] \
  [get_bd_pins axi_interconnect_0/S00_ACLK] \
  [get_bd_pins axi_interconnect_0/M00_ACLK] \
  [get_bd_pins axi_pc_0/aclk] \
  [get_bd_pins bitnet_ip_top_0/aclk]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
  [get_bd_pins processing_system7_0/M_AXI_GP0_ARESETN] \
  [get_bd_pins axi_interconnect_0/ARESETN] \
  [get_bd_pins axi_interconnect_0/S00_ARESETN] \
  [get_bd_pins axi_interconnect_0/M00_ARESETN] \
  [get_bd_pins axi_pc_0/aresetn] \
  [get_bd_pins bitnet_ip_top_0/aresetn]

assign_bd_address
# Auto-assign picks 0x4000_0000; offset is read-only — recreate segment at capstone base.
delete_bd_objs [get_bd_addr_segs /bitnet_ip_top_0/S_AXI/reg0]
create_bd_addr_seg -range 4K -offset 0x43C00000 \
  [get_bd_addr_spaces processing_system7_0/Data] \
  [get_bd_addr_segs /bitnet_ip_top_0/S_AXI/reg0] \
  reg0

validate_bd_design
save_bd_design

set bd_file [get_files design_1.bd]
make_wrapper -files $bd_file -top
set wrapper_glob [glob -nocomplain [file join $proj_dir bitnet_zybo.gen sources_1 bd design_1 hdl design_1_wrapper.v]]
if {$wrapper_glob eq ""} {
  error "Could not find design_1_wrapper.v under $proj_dir/bitnet_zybo.gen"
}
add_files -norecurse $wrapper_glob
update_compile_order -fileset sources_1
set_property top design_1_wrapper [current_fileset]

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
  error "Synthesis failed"
}

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
  error "Implementation/bitstream failed"
}

set bit [file join $proj_dir bitnet_zybo.runs impl_1 design_1_wrapper.bit]
if {[file exists $bit]} {
  file copy -force $bit [file join $out_dir bitnet_zybo.bit]
  puts "Bitstream: [file join $out_dir bitnet_zybo.bit]"
} else {
  error "Bitstream file not found"
}

set xsa [file join $out_dir bitnet_zybo.xsa]
write_hw_platform -fixed -force -include_bit -file $xsa
puts "Hardware export: $xsa"
