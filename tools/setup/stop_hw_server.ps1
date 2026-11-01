# Stop stray Xilinx hw_server / cs_server (use before a fresh make list-hw).
$names = @("hw_server", "cs_server", "xvdb_server")
foreach ($n in $names) {
    Get-Process -Name $n -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
Write-Host "Stopped Xilinx lab server processes (if any were running)."
