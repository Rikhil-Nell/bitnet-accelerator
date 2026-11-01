# Elaborate + synthesize bitnet_ip_top for utilization sanity (no bitstream yet).
set repo_root [file normalize [file join [info script] .. .. ..]]
set rtl_dir   [file join $repo_root bitnet-fpga-accelerator rtl des]
set out_dir   [file join $repo_root build vivado]
file mkdir $out_dir

set_part xc7z010clg400-1

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

read_verilog $rtl_files
synth_design -top bitnet_ip_top -part xc7z010clg400-1 -flatten_hierarchy rebuilt

report_utilization -file [file join $out_dir utilization.rpt]
report_timing_summary -file [file join $out_dir timing.rpt]

puts "Synthesis finished. Reports in $out_dir"
