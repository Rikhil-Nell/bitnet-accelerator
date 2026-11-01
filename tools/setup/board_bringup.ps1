# Zybo end-to-end: JTAG program PL + XSDB run PS (+ optional UART listen).
# Requires USB Serial Converter B OK (see docs/ZYBO_JTAG.md).
param(
    [string]$Com = "",
    [switch]$Listen
)

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
. "$Root\tools\setup\env.ps1"

& "$Root\tools\setup\stage_board_assets.ps1"
& "$Root\tools\setup\diagnose_zybo_usb.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $Com) {
    $ports = Get-CimInstance Win32_SerialPort -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'USB Serial|FTDI|Digilent' }
    if ($ports) {
        $Com = ($ports | Select-Object -First 1).DeviceID
        Write-Host "Auto-selected UART: $Com"
    } else {
        $Com = "COM3"
        Write-Host "No COM auto-detect; default $Com (override: -Com COMx)"
    }
}

function Invoke-VivadoTcl {
    param([string]$Script, [string]$LogName)
    $log = Join-Path $Root "build/vivado/$LogName"
    New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
    & vivado -mode batch -source $Script -log $log
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "== list-hw =="
if (Get-Command make -ErrorAction SilentlyContinue) {
    make list-hw
} else {
    Invoke-VivadoTcl -Script "fpga/scripts/list_hw_targets.tcl" -LogName "list_hw.log"
}

$log = Get-Content "$Root\build\vivado\list_hw.log" -Raw
if ($log -notmatch "targets on localhost:3121: \S") {
    Write-Host ""
    Write-Host "No JTAG target visible. See docs/ZYBO_JTAG.md" -ForegroundColor Yellow
    exit 1
}

Write-Host "== program PL =="
if (Get-Command make -ErrorAction SilentlyContinue) {
    make program
} else {
    Invoke-VivadoTcl -Script "fpga/scripts/program_zybo.tcl" -LogName "program.log"
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== run PS ELF (XSDB) ==" -ForegroundColor Green
& "$Root\tools\setup\run_ps_xsdb.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Listen) {
    Write-Host "== UART listen =="
    python "$Root\tools\host\uart_board_test.py" --port $Com
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "PS app should be running. UART listen:"
Write-Host "  python tools/host/uart_board_test.py --port $Com"
Write-Host 'Expected: PASS: system_8x8 on PL accelerator (115200)'
