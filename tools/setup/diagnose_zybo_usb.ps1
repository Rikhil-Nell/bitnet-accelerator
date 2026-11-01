# Report Zybo FTDI (6010) status and next steps for JTAG.
$ErrorActionPreference = "Continue"
$ftdi = Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -match 'VID_0403&PID_6010' }
if (-not $ftdi) {
    Write-Host "No FTDI 6010 device found. Plug Zybo PROG micro-USB (SW7 on)."
    exit 1
}
$ftdi | Format-Table Status, FriendlyName, InstanceId -AutoSize
$b = $ftdi | Where-Object { $_.FriendlyName -match 'Converter B' } | Select-Object -First 1
if ($b -and $b.Status -ne 'OK') {
    Write-Host ""
    Write-Host "Converter B is not OK (Problem=$($b.Problem))." -ForegroundColor Yellow
    switch ($b.Problem) {
        'CM_PROB_NEED_RESTART' {
            $boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
            Write-Host "  -> Problem code 14: driver change pending REBOOT (LastBootUpTime: $boot)."
            Write-Host "  -> Full shutdown/restart (not Sleep), replug PROG USB, then: . tools/setup/post_reboot_verify.ps1"
        }
        default {
            Write-Host "  -> Install Digilent Adept; Device Manager: uninstall Converter B + delete driver; replug."
        }
    }
    Write-Host "  -> docs/ZYBO_JTAG.md"
    exit 1
}
Write-Host "FTDI looks OK. Run: make list-hw"
exit 0
