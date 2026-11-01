set repo_root [file normalize [file join [info script] .. .. ..]]
set proj_dir  [file join $repo_root build bitnet_zybo]
set out_dir   [file join $repo_root build vivado]
set xsa       [file join $out_dir bitnet_zybo.xsa]

open_project [file join $proj_dir bitnet_zybo.xpr]
open_run impl_1
write_hw_platform -fixed -force -include_bit -file $xsa
puts "Hardware export: $xsa"
