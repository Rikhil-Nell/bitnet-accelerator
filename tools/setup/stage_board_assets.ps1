# Copy bitstream / ps7_init to D:\xilinx-tools (no spaces — safer for Tcl/XSDB).
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Dst = "D:\xilinx-tools"
New-Item -ItemType Directory -Force -Path $Dst | Out-Null

$bit = Join-Path $Root "build\vivado\bitnet_zybo.bit"
$xsa = Join-Path $Root "build\vivado\bitnet_zybo.xsa"
if (Test-Path $bit) { Copy-Item -Force $bit (Join-Path $Dst "bitnet_zybo.bit") }
if (Test-Path $xsa) { Copy-Item -Force $xsa (Join-Path $Dst "bitnet_zybo.xsa") }

$ps7 = Get-ChildItem -Path (Join-Path $Root "build\bitnet_zybo") -Recurse -Filter "ps7_init.tcl" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($ps7) {
    Copy-Item -Force $ps7.FullName (Join-Path $Dst "ps7_init.tcl")
    Write-Host "Staged: $Dst\ps7_init.tcl"
} else {
    Write-Warning "ps7_init.tcl not found under build/bitnet_zybo (run make bitstream)"
}
Write-Host "Staged board assets under $Dst"
