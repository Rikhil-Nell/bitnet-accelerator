# Create Vitis platform + standalone app for system_8x8_test (batch xsct).
set repo_root [file normalize [file join [info script] .. .. ..]]
set xsa [file join $repo_root build vivado bitnet_zybo.xsa]
set ws [file join $repo_root build vitis]
set app_name system_8x8_test
set ps_src [file join $repo_root fpga ps system_8x8_test]

if {![file exists $xsa]} {
  error "Missing $xsa — run make bitstream or make export-xsa"
}

file mkdir $ws
cd $ws

platform create -name bitnet_platform -hw $xsa -proc ps7_cortexa9_0 -os standalone
platform write
platform generate

app create -name $app_name -platform bitnet_platform -domain standalone_domain -template {Empty Application}
set app_dir [file join $ws $app_name src]

foreach f [glob -nocomplain [file join $ps_src *]] {
  file copy -force $f $app_dir
}

app config -name $app_name build-config release
app build -name $app_name

set elf [file join $ws $app_name build $app_name.elf]
if {![file exists $elf]} {
  error "Build failed: $elf not found"
}
puts "Built PS app: $elf"
