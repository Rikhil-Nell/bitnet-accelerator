# Opens AMD download pages and prints D: install checklist.
# Vivado web installer requires AMD login in the browser — cannot be fully automated without your account.

$downloads = "D:\xilinx-tools\downloads"
New-Item -ItemType Directory -Force -Path $downloads | Out-Null

Write-Host @"

=== BitNet Zybo toolchain (manual step required) ===
1. Log in at AMD Adaptive downloads (browser opening).
2. Save the Windows Web Installer to: $downloads
3. Run installer -> Install to D:\Xilinx -> Zynq-7000 only.
4. Add board repo: D:\xilinx-tools\vivado-boards\new\board_files
5. Verify: . tools\setup\env.ps1 ; vivado -version

Optional Ross MCP: https://www.amd.com/en/support/downloads/ross-agentic-ai.html

"@

Start-Process "https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools/2025-1.html"
Start-Process "https://www.amd.com/en/support/downloads/ross-agentic-ai.html"
