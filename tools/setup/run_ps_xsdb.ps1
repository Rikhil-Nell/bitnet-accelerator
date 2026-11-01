# XSDB via the official xsdb.bat launcher with a minimal PATH
# (full Vitis settings64.bat hits the Windows PATH length limit; the unwrapped
# xsdb.exe cannot find the xsdb Tcl package).
param([string]$Script = "fpga/scripts/run_ps_zybo.tcl")
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$xsdb = "D:\Xilinx\2026.1\Vitis\bin\xsdb.bat"
if (-not (Test-Path $xsdb)) {
    Write-Error "xsdb not found: $xsdb"
}
Set-Location $Root
$extra = @()
if ($env:BITNET_ELF) { $extra += "set BITNET_ELF=$($env:BITNET_ELF)" }
# Second core entry point, if the ELF has one (fpga/ps/bitnet_zybo/cpu1.S)
$nm = "D:\Xilinx\2026.1\gnu\aarch32\nt\gcc-arm-none-eabi\bin\arm-none-eabi-nm.exe"
if ($env:BITNET_ELF -and $env:BITNET_ELF -ne "skip" -and (Test-Path $nm)) {
    $sym = & $nm $env:BITNET_ELF 2>$null | Select-String " T cpu1_entry$"
    if ($sym) { $extra += "set BITNET_CPU1_ENTRY=0x$(($sym.Line -split ' ')[0])" }
}
if ($env:BITNET_DATA) { $extra += "set BITNET_DATA=$($env:BITNET_DATA)" }
if ($env:BITNET_DATA_ADDR) { $extra += "set BITNET_DATA_ADDR=$($env:BITNET_DATA_ADDR)" }
if ($env:BITNET_JTAG_HZ) { $extra += "set BITNET_JTAG_HZ=$($env:BITNET_JTAG_HZ)" }
$prefix = ($extra -join "&& ")
if ($prefix) { $prefix += "&& " }
cmd /c "${prefix}set PATH=C:\Windows\System32;C:\Windows;D:\Xilinx\2026.1\Vitis\bin&& set VITIS_EMBEDDED_INSTALL=true&& $xsdb $Script"
exit $LASTEXITCODE
