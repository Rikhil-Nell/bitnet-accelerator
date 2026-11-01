# XSDB: system reset of the Zynq PS through the DAP (recovers an A9 debug port that
# reports "APB AP transaction error"). The PL bitstream is lost; run make program after.
connect
after 1500
foreach t [targets] { puts "  $t" }
if {[catch {targets -set -filter {name =~ "DAP*"}} msg]} { puts "ERROR: no DAP target: $msg"; exit 1 }
if {[catch {rst -srst} msg]} { puts "ERROR: rst -srst: $msg"; exit 1 }
after 2000
puts "after reset:"
foreach t [targets] { puts "  $t" }
exit 0
