# Program PL (optional), load weights if BITNET_SKIP_LOAD is unset, start bitlinear_server on the A9.
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$elf = if ($env:BITNET_ELF) { $env:BITNET_ELF } else {
    "D:/xilinx-tools/bitnet_vitis_ws_clean/bitlinear_server/build/bitlinear_server.elf"
}
if (-not (Test-Path $elf)) {
    Write-Error "Missing $elf — run: make bitlinear-ps-app"
}
$env:BITNET_ELF = $elf
if (-not $env:BITNET_SKIP_LOAD) {
    $bin = if ($env:BITNET_WEIGHTS_BIN) { $env:BITNET_WEIGHTS_BIN } else { "D:/xilinx-tools/bitnet_weights.bin" }
    if (Test-Path $bin) {
        $env:BITNET_DATA = $bin
        $env:BITNET_DATA_ADDR = if ($env:BITNET_DATA_ADDR) { $env:BITNET_DATA_ADDR } else { "0x10000000" }
        $env:BITNET_JTAG_HZ = if ($env:BITNET_JTAG_HZ) { $env:BITNET_JTAG_HZ } else { "30000000" }
    } else {
        Write-Warning "No $bin — assuming weights already in DDR (BITNET_SKIP_LOAD=1 next time)"
        Remove-Item Env:BITNET_DATA -ErrorAction SilentlyContinue
    }
}
Set-Location $Root
& "$Root\tools\setup\run_ps_xsdb.ps1"
exit $LASTEXITCODE
