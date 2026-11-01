# Simulate gemv_top in Vivado xsim against Python-generated vectors.
#   powershell -File fpga/scripts/sim_gemv.ps1 [-Stall 25] [-Seed 1]
param([int]$Stall = 25, [int]$Seed = 1)
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
. "$Root\tools\setup\env.ps1" | Out-Null

$Out = Join-Path $Root "build\sim\gemv"
New-Item -ItemType Directory -Force $Out | Out-Null
Set-Location $Out

python "$Root\rtl\tb\gen_vectors.py" $Out --seed $Seed
if ($LASTEXITCODE) { exit 1 }

$rtl = @("axi3_rd.v", "axi3_wr.v", "gemv_unit.v", "gemv_top.v") | ForEach-Object { Join-Path $Root "rtl\$_" }
xvlog --nolog $rtl | Out-Null
if ($LASTEXITCODE) { xvlog $rtl; exit 1 }
xvlog --nolog -sv (Join-Path $Root "rtl\tb\tb_gemv_top.sv") | Out-Null
if ($LASTEXITCODE) { xvlog -sv (Join-Path $Root "rtl\tb\tb_gemv_top.sv"); exit 1 }
xelab --nolog -s tb_gemv tb_gemv_top -timescale 1ns/1ps | Out-Null
if ($LASTEXITCODE) { xelab -s tb_gemv tb_gemv_top; exit 1 }

$log = Join-Path $Out "xsim.log"
Set-Content -Path stall.txt -Value $Stall
xsim tb_gemv --runall --log xsim.log | Out-Null
Get-Content $log | Select-String -Pattern "^(case|FAIL|TEST|ERROR)"
if (Select-String -Path $log -Pattern "TEST PASSED" -Quiet) { exit 0 } else { exit 1 }
