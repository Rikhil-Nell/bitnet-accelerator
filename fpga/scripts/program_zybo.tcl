# Program bitstream to first connected target (Zybo via hw_server).
set bit_staged {D:/xilinx-tools/bitnet_zybo.bit}
set bit [file normalize [file join [info script] .. .. .. build vivado bitnet_zybo.bit]]
if {[file exists $bit_staged]} {
  set bit [file normalize $bit_staged]
}
if {![file exists $bit]} {
  error "Bitstream not found: $bit (run make bitstream first)"
}

open_hw_manager
connect_hw_server
set hw_targets [get_hw_targets -quiet]
if {[llength $hw_targets] == 0} {
  after 3000
  set hw_targets [get_hw_targets -quiet]
}
if {[llength $hw_targets] == 0} {
  puts "DIAG: hw_server OK but no targets. On Zybo, check FTDI 'USB Serial Converter B' in Device Manager (Digilent/JTAG channel)."
  puts "See docs/ZYBO_JTAG.md"
  error "No JTAG target on localhost."
}
open_hw_target [lindex $hw_targets 0]
set devs [get_hw_devices -quiet]
if {[llength $devs] == 0} {
  error "Target open but no hw_device (refresh cable / use hw_server)."
}
set dev [lindex [get_hw_devices -quiet xc7z*] 0]
if {$dev eq ""} {
  error "No xc7z* FPGA device on target (found: $devs)."
}
current_hw_device $dev
set cfg [current_hw_cfgmem]
refresh_hw_device -update_hw_probes false $dev
set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
puts "Programmed $bit to $dev"
