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
if ($env:BITNET_DATA) { $extra += "set BITNET_DATA=$($env:BITNET_DATA)" }
if ($env:BITNET_DATA_ADDR) { $extra += "set BITNET_DATA_ADDR=$($env:BITNET_DATA_ADDR)" }
if ($env:BITNET_JTAG_HZ) { $extra += "set BITNET_JTAG_HZ=$($env:BITNET_JTAG_HZ)" }
$prefix = ($extra -join "&& ")
if ($prefix) { $prefix += "&& " }
cmd /c "${prefix}set PATH=C:\Windows\System32;C:\Windows;D:\Xilinx\2026.1\Vitis\bin&& set VITIS_EMBEDDED_INSTALL=true&& $xsdb $Script"
exit $LASTEXITCODE
