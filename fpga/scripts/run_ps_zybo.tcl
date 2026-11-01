# XSDB batch: init PS, optionally load a data file into DDR, download + run an ELF.
# Env: BITNET_ELF (path, or "skip" for data-only), BITNET_DATA + BITNET_DATA_ADDR (optional).
set repo [file normalize [file join [info script] .. .. ..]]
set ps7_tcl [file join $repo build vivado ps7_init.tcl]
if {![info exists ::env(BITNET_ELF)] || $::env(BITNET_ELF) eq ""} {
  puts "ERROR: set BITNET_ELF (make run-ps ELF=path/to/app.elf)"
  exit 1
}
set skip_elf [expr {$::env(BITNET_ELF) eq "skip"}]
if {!$skip_elf} {
  set elf [file normalize $::env(BITNET_ELF)]
  if {![file exists $elf]} {
    puts "ERROR: ELF not found: $elf"
    exit 1
  }
}

puts "XSDB: connect"
connect
after 2000
set tgts [targets]
if {[llength $tgts] == 0} {
  puts "ERROR: No debug targets. Fix USB Serial Converter B (docs/ZYBO_JTAG.md)"
  exit 1
}

set apu [lindex [targets -filter {name =~ "*A9*#0*"}] 0]
if {$apu eq ""} {
  set apu [lindex [targets -filter {name =~ "*Cortex-A9*"}] 0]
}
if {$apu eq ""} {
  puts "Available targets:"
  foreach t [targets] { puts "  $t" }
  puts "ERROR: Could not select Cortex-A9 #0"
  exit 1
}

targets -set $apu
catch {stop}
if {![file exists $ps7_tcl]} {
  puts "ERROR: Missing $ps7_tcl (run make bitstream; it copies ps7_init.tcl there)"
  exit 1
}
puts "XSDB: ps7_init from BD export"
source $ps7_tcl
ps7_init
# Enables PS->PL level shifters; without it the first AXI GP0 access hangs the A9.
ps7_post_config

if {[info exists ::env(BITNET_DATA)] && $::env(BITNET_DATA) ne ""} {
  set data [file normalize $::env(BITNET_DATA)]
  set addr $::env(BITNET_DATA_ADDR)
  set mb [expr {[file size $data] / 1048576.0}]
  puts [format "XSDB: dow -data %s @ %s (%.1f MiB)" $data $addr $mb]
  if {[info exists ::env(BITNET_JTAG_HZ)] && $::env(BITNET_JTAG_HZ) ne ""} {
    if {[catch {jtag targets 1; jtag frequency $::env(BITNET_JTAG_HZ)} msg]} {
      puts "XSDB: jtag frequency not changed ($msg)"
    } else {
      puts "XSDB: jtag frequency [jtag frequency]"
    }
    targets -set $apu
  }
  set t0 [clock milliseconds]
  dow -data $data $addr
  set dt [expr {([clock milliseconds] - $t0) / 1000.0}]
  puts [format "XSDB: data loaded in %.1f s (%.2f MiB/s)" $dt [expr {$mb / $dt}]]
}

if {!$skip_elf} {
  puts "XSDB: dow $elf"
  dow $elf
  puts "XSDB: continue"
  con
} else {
  puts "XSDB: BITNET_ELF=skip (data-only session)"
}
exit 0
