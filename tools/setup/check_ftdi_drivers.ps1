# Refuse JTAG access when FTDI bus and port drivers differ in version.
# A 2024 ftdibus.sys with a 2017 ftser2k.sys bugchecked this PC twice (0x139, 0x50)
# as soon as hw_server.exe opened the Zybo. See docs/ZYBO_JTAG.md.
$root = if ($env:SystemRoot) { $env:SystemRoot } elseif ($env:windir) { $env:windir } else { 'C:\Windows' }
# 32-bit hosts (e.g. GnuWin make) get System32 redirected to SysWOW64.
$sys = if ([Environment]::Is64BitProcess) { 'System32' } else { 'Sysnative' }
$drv = "$root\$sys\drivers"
$bus = Get-Item "$drv\ftdibus.sys" -ErrorAction SilentlyContinue
$ser = Get-Item "$drv\ftser2k.sys" -ErrorAction SilentlyContinue
if (-not $bus) {
    Write-Host "FTDI check: ftdibus.sys not found (no FTDI driver installed)."
    exit 2
}
$busVer = [version]($bus.VersionInfo.FileVersion -replace '[^0-9.]', '')
Write-Host ("FTDI check: ftdibus.sys {0}" -f $busVer)
if (-not $ser) {
    Write-Host "FTDI check: FAIL - ftser2k.sys missing. Plug in the Zybo so Windows installs the port driver, then re-run."
    exit 1
}
if ($ser) {
    $serVer = [version]($ser.VersionInfo.FileVersion -replace '[^0-9.]', '')
    Write-Host ("FTDI check: ftser2k.sys {0}" -f $serVer)
    if ($busVer.Major -ne $serVer.Major -or $busVer.Minor -ne $serVer.Minor -or $busVer.Build -ne $serVer.Build) {
        Write-Host "FTDI check: FAIL - bus/port driver versions differ. Install the FTDI CDM package and reboot before using JTAG."
        exit 1
    }
}
Write-Host "FTDI check: OK"
exit 0
