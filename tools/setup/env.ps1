# Source before any Xilinx tool invocation (new shell every agent command):
#   . "$PSScriptRoot\env.ps1"

$ErrorActionPreference = "Stop"

function Import-XilinxSettingsBat {
    param([Parameter(Mandatory = $true)][string]$BatPath)
    if (-not (Test-Path $BatPath)) {
        throw "Settings batch file not found: $BatPath"
    }
    $envBlock = cmd.exe /c "call `"$BatPath`" >nul 2>&1 && set"
    foreach ($line in $envBlock) {
        if ($line -match '^(.*?)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}

$XilinxRoot = if ($env:XILINX_ROOT) { $env:XILINX_ROOT } else { "D:\Xilinx" }
$XilinxVersion = if ($env:XILINX_VERSION) { $env:XILINX_VERSION } else { "2026.1" }

$candidates = @(
    (Join-Path $XilinxRoot "$XilinxVersion\Vivado\settings64.ps1")
    (Join-Path $XilinxRoot "Vivado\$XilinxVersion\settings64.ps1")
    (Join-Path $XilinxRoot "$XilinxVersion\Vivado\settings64.bat")
    (Join-Path $XilinxRoot "$XilinxVersion\Vitis\settings64.bat")
)

$loaded = $false
foreach ($path in $candidates) {
    if (-not (Test-Path $path)) { continue }
    if ($path.EndsWith(".ps1")) {
        . $path
        $loaded = $true
        break
    }
    if ($path.EndsWith(".bat")) {
        Import-XilinxSettingsBat -BatPath $path
        $loaded = $true
        break
    }
}

if (-not $loaded) {
    Write-Warning "Xilinx settings not found under $XilinxRoot (version $XilinxVersion)."
    return
}

$boardRepo = "D:/xilinx-tools/vivado-boards/new/board_files"
$env:BOARD_REPO_PATHS = $boardRepo
# Vivado Tcl param (used by batch scripts)
$env:VIVADO_BOARD_REPO = $boardRepo
$env:BITNET_PART = "xc7z010clg400-1"
$env:BITNET_BOARD_PART = "digilentinc.com:zybo-z7-10:part0:1.2"

Write-Host ("Xilinx environment loaded. XILINX_VIVADO=" + $env:XILINX_VIVADO)
