# Placeholder — filled in after Vivado 2026.1 install is verified.
# Usage: vivado -mode batch -source fpga/scripts/build_bitstream.tcl

puts "ERROR: Bitstream flow not wired yet."
puts "Next steps:"
puts "  1) Create BD: ZYNQ7 + AXI GP0 + bitnet AXI-Lite slave"
puts "  2) Add constraints from D:/xilinx-tools/digilent-xdc/Zybo-Z7-Master.xdc"
puts "  3) set_property board_part digilent.com:zybo-z7-10:part0:1.0"
exit 1
