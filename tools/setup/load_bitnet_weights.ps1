# Load packed BitNet weights into board DDR via XSDB (JTAG). Slow (~497 MiB); use 30 MHz JTAG if supported.
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$bin = if ($env:BITNET_WEIGHTS_BIN) { $env:BITNET_WEIGHTS_BIN } else { "D:/xilinx-tools/bitnet_weights.bin" }
$addr = if ($env:BITNET_DATA_ADDR) { $env:BITNET_DATA_ADDR } else { "0x10000000" }
if (-not (Test-Path $bin)) {
    Write-Error "Missing $bin — run: make pack-bitnet-weights"
}
$env:BITNET_DATA = $bin
$env:BITNET_DATA_ADDR = $addr
$env:BITNET_JTAG_HZ = if ($env:BITNET_JTAG_HZ) { $env:BITNET_JTAG_HZ } else { "30000000" }
$env:BITNET_ELF = "skip"
Set-Location $Root
& "$Root\tools\setup\run_ps_xsdb.ps1"
exit $LASTEXITCODE
