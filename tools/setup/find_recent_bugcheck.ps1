# List recent kernel crash / unexpected reboot markers (read-only Event Log).
# DCOM 10016 (Security Center) is unrelated noise — look for 1001 BugCheck or 41 Kernel-Power.
$since = (Get-Date).AddDays(-3)
Write-Host "System log since $since (BugCheck / Kernel-Power / WHEA):"
Get-WinEvent -FilterHashtable @{
    LogName = 'System'
    StartTime = $since
    Id = 1001, 41, 6008
} -ErrorAction SilentlyContinue |
    Select-Object -First 15 TimeCreated, Id, ProviderName, Message |
    Format-List

Write-Host ""
Write-Host "Tip: Event 1001 includes stop code and faulting driver; paste that for JTAG/BSOD triage."
