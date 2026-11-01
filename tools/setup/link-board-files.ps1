# Optional: junction Digilent boards into the Vivado install tree (requires Admin).
# Easier: set BOARD_REPO_PATHS in env.ps1 (already done) and add the same path in
# Vivado GUI: Settings -> Vivado Store -> Board Repository.
param(
    [string]$VivadoVersion = "2026.1"
)

$xr = if ($env:XILINX_ROOT) { $env:XILINX_ROOT } else { "D:\Xilinx" }
$target = Join-Path $xr "$VivadoVersion\Vivado\data\boards\board_files"
$work = if ($env:BITNET_WORK) { $env:BITNET_WORK } else { "D:\xilinx-tools" }
$source = if ($env:BITNET_BOARD_FILES) { $env:BITNET_BOARD_FILES } else { Join-Path $work "vivado-boards\new\board_files" }

if (-not (Test-Path $source)) {
    Write-Error "Missing board files at $source"
}

New-Item -ItemType Directory -Force -Path $target | Out-Null

Get-ChildItem $source -Directory | ForEach-Object {
    $link = Join-Path $target $_.Name
    if (Test-Path $link) {
        Write-Host "Skip (exists): $link"
    } else {
        New-Item -ItemType Junction -Path $link -Target $_.FullName | Out-Null
        Write-Host "Linked: $($_.Name)"
    }
}

Write-Host 'Done. Restart Vivado and select Zybo Z7-10 board.'
