# Run once after reboot + Zybo PROG USB replugged.
# Chains Vivado + XSDB; if the PC BSODs on JTAG, run steps in docs/ZYBO_JTAG.md manually instead.
param(
    [string]$Com = "",
    [switch]$AllowVivado
)

if (-not $AllowVivado) {
    Write-Host "Refusing to run Vivado/XSDB automatically (use -AllowVivado after make list-hw is stable)."
    Write-Host "Manual: make list-hw -> make program -> make run-ps -> make uart-listen COM=COMx"
    exit 2
}

$ErrorActionPreference = "Continue"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
$logDir = Join-Path $Root "build/capstone"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "post_reboot.log"
Start-Transcript -Path $log -Append | Out-Null

Write-Host "=== post-reboot verify $(Get-Date -Format o) ==="

if ($Com) {
    & "$Root\tools\setup\board_bringup.ps1" -Com $Com -Listen
} else {
    & "$Root\tools\setup\board_bringup.ps1" -Listen
}
$rc = $LASTEXITCODE

$env:Path = "$env:SystemRoot\System32;$env:SystemRoot;$env:Path"
python "$Root\tools\eval\capstone_check.py"
Stop-Transcript | Out-Null
exit $rc
