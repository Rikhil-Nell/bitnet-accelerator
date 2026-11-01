# Batch RTL sim (xsim) when Icarus is unavailable.
if {[info exists env(BITNET_REPO_ROOT)] && [string length $env(BITNET_REPO_ROOT)] > 0} {
  set repo_root [file normalize $env(BITNET_REPO_ROOT)]
} else {
  set repo_root [file normalize [file join [info script] .. .. ..]]
}
set rtl_dir   [file join $repo_root bitnet-fpga-accelerator rtl des]
set tb_dir    [file join $repo_root bitnet-fpga-accelerator rtl tb]
set proj_dir  "D:/xilinx-tools/bitnet_xsim"
set log_dir   [file join $repo_root build sim]
file mkdir $proj_dir
file mkdir $log_dir

proc run_tb {top rtl_files tb_file log_name pass_marker} {
  global proj_dir log_dir
  set run_dir [file join $proj_dir $top]
  file mkdir $run_dir
  create_project -force $top $run_dir -part xc7z010clg400-1
  foreach f [concat $rtl_files [list $tb_file]] {
    add_files -norecurse [file normalize $f]
  }
  set_property top $top [get_filesets sources_1]
  set_property top $top [get_filesets sim_1]
  update_compile_order -fileset sources_1
  update_compile_order -fileset sim_1
  set_property -name {xsim.simulate.runtime} -value {all} -objects [get_filesets sim_1]
  launch_simulation -simset sim_1
  run all
  close_sim -force
  set sim_log [file join $run_dir $top.sim sim_1 behav xsim simulate.log]
  if {![file exists $sim_log]} {
    error "Missing xsim log: $sim_log"
  }
  set fd [open $sim_log r]
  set body [read $fd]
  close $fd
  set out [file join $log_dir $log_name]
  set fout [open $out w]
  puts -nonewline $fout $body
  close $fout
  if {[string first $pass_marker $body] < 0} {
    error "$top: marker '$pass_marker' not found (see $out)"
  }
  puts "PASS: $top (xsim)"
}

set decoder_rtl [list [file join $rtl_dir ternary_decoder.v]]
run_tb ternary_decoder_tb $decoder_rtl [file join $tb_dir ternary_decoder_tb.v] xsim_decoder.log "TEST COMPLETED"

set mac_rtl [list \
  [file join $rtl_dir ternary_decoder.v] \
  [file join $rtl_dir ternary_mac.v]]
run_tb ternary_mac_tb $mac_rtl [file join $tb_dir ternary_mac_tb.v] xsim_mac.log "TEST COMPLETED"

set accel_rtl [list \
  [file join $rtl_dir ternary_decoder.v] \
  [file join $rtl_dir ternary_mac.v] \
  [file join $rtl_dir accumulator.v] \
  [file join $rtl_dir activation_memory.v] \
  [file join $rtl_dir weight_memory.v] \
  [file join $rtl_dir output_memory.v] \
  [file join $rtl_dir controller_fsm.v] \
  [file join $rtl_dir bitnet_accelerator.v]]
run_tb bitnet_accelerator_tb $accel_rtl [file join $tb_dir bitnet_accelerator_tb.v] xsim_accelerator.log "TEST PASSED"

set ip_top_rtl [concat $accel_rtl [list \
  [file join $rtl_dir control_registers.v] \
  [file join $rtl_dir bitnet_axi_lite.v] \
  [file join $rtl_dir bitnet_ip_top.v]]]
run_tb bitnet_ip_top_tb $ip_top_rtl [file join $tb_dir bitnet_ip_top_tb.v] xsim_ip_top.log "TEST PASSED"

puts "All xsim RTL smoke tests finished."
