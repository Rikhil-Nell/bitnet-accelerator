# Run a Vitis Python script with the Vitis environment (settings64.bat).
# Goes through a temporary .bat because the repo path has a space, which nested
# make -> sh -> cmd quoting cannot carry reliably.
#   powershell -File tools/setup/vitis_run.ps1 -Script fpga/scripts/build_ps_platform.py -App gemv_bench
param(
    [Parameter(Mandatory = $true)][string]$Script,
    [string]$App = "gemv_bench"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$settings = "D:\Xilinx\2026.1\Vitis\settings64.bat"
$bat = Join-Path $env:TEMP "bitnet_vitis_run.bat"
@"
@echo off
call "$settings" >nul 2>&1
set VITIS_EMBEDDED_INSTALL=true
set BITNET_PS_APP=$App
cd /d "$Root"
vitis -s "$Script"
"@ | Set-Content -Encoding ASCII $bat
# vitis.dispose() leaves the Vitis Java server alive and it keeps the workspace
# locked, so stop it before and after each run. Its path is reported with forward
# slashes ("D:/Xilinx/..."), hence the regex. hw_server (JTAG) is left alone.
# make may launch the 32-bit PowerShell, where Get-Process cannot read a 64-bit
# process's Path, so use WMI, which can.
function Stop-VitisServer {
    Get-CimInstance Win32_Process -Filter "Name = 'java.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ExecutablePath -match '^D:[\\/]Xilinx[\\/]' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Stop-VitisServer
cmd /c $bat
$rc = $LASTEXITCODE
Stop-VitisServer
exit $rc
