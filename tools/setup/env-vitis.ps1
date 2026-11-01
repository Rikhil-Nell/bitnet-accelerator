# Vivado + Vitis (embedded) — use before xsct/buildplatform/vitis-run.
. "$PSScriptRoot\env.ps1"

$vitisBat = "D:\Xilinx\2026.1\Vitis\settings64.bat"
if (-not (Test-Path $vitisBat)) {
    Write-Warning "Vitis settings not found: $vitisBat"
    return
}

function Import-XilinxSettingsBat {
    param([Parameter(Mandatory = $true)][string]$BatPath)
    $envBlock = cmd.exe /c "call `"$BatPath`" >nul 2>&1 && set"
    foreach ($line in $envBlock) {
        if ($line -match '^(.*?)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}

Import-XilinxSettingsBat -BatPath $vitisBat
$env:VITIS_EMBEDDED_INSTALL = "true"
Write-Host ("Vitis environment loaded. XILINX_VITIS=" + $env:XILINX_VITIS)
