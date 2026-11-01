set_param board.repoPaths [list D:/xilinx-tools/vivado-boards/new/board_files]
foreach bp [get_board_parts] {
  if {[string match -nocase *zybo* $bp]} {
    puts $bp
  }
}
