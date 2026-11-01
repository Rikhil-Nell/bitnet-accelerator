open_hw_manager
connect_hw_server
set servers [get_hw_servers -quiet]
puts "hw_servers: $servers"
foreach srv $servers {
  set tgts [get_hw_targets -of_objects $srv -quiet]
  puts "targets on $srv: $tgts"
}
if {[llength [get_hw_targets -quiet]] > 0} {
  open_hw_target [lindex [get_hw_targets] 0]
  puts "devices: [get_hw_devices]"
}
exit 0
