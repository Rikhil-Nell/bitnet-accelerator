# Clear Vitis workspace peer lock (close Vitis IDE first if open).
param(
    [string]$Ws = "D:\xilinx-tools\bitnet_vitis_ws_clean"
)
$ide = Join-Path $Ws "_ide"
foreach ($f in @(".peers.ini", "workspace_journal.py")) {
    $p = Join-Path $ide $f
    if (Test-Path $p) {
        Remove-Item $p -Force
        Write-Host "Removed $p"
    }
}
Write-Host "Retry: make ps-app"
