# Interpret build/vivado/list_hw.log without starting Vivado.
param(
    [string]$Log = ""
)

$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $Log) { $Log = Join-Path $Root "build\vivado\list_hw.log" }

if (-not (Test-Path $Log)) {
    Write-Host "MISSING: $Log (run make list-hw-safe locally)"
    exit 2
}

$text = Get-Content $Log -Raw
if ($text -notmatch "targets on localhost:3121:") {
    if ($text -match "open_hw_manager" -and $text -notmatch "connect_hw_server") {
        Write-Host "INCOMPLETE: log stops before connect_hw_server (crash or kill?)"
        exit 3
    }
    Write-Host "INCOMPLETE: no targets line in log"
    exit 3
}

foreach ($line in (Get-Content $Log)) {
    if ($line -match "^targets on localhost:3121:\s*(.+)$") {
        $t = $Matches[1].Trim()
        if ($t) {
            Write-Host "PASS: JTAG target visible: $t"
            exit 0
        }
        Write-Host "EMPTY: hw_server OK but no hw_targets (Digilent Adept? power/cable?)"
        exit 1
    }
    if ($line -match "^targets on localhost:3121:\s*$") {
        Write-Host "EMPTY: hw_server OK but no hw_targets (Digilent Adept? power/cable?)"
        exit 1
    }
}
Write-Host "UNKNOWN: could not parse targets line"
exit 4
