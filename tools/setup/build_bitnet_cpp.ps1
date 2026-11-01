# Build Microsoft BitNet (bitnet.cpp / llama-cli) with VS ClangCL + bundled CMake.
$ErrorActionPreference = "Stop"
$BitNet = if ($env:BITNET_CPP_ROOT) { $env:BITNET_CPP_ROOT } else { "D:\src\BitNet" }
$cmake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$vc = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$model = if ($env:BITNET_MODEL_DIR) { $env:BITNET_MODEL_DIR } else { "models/BitNet-b1.58-2B-4T" }
if (-not (Test-Path (Join-Path $BitNet (Join-Path $model "ggml-model-i2_s.gguf")))) {
    Write-Error "Missing GGUF under $BitNet\$model - create junction to C:\AI\models\BitNet-b1.58-2B-4T-gguf"
}
$cmdLine = "call `"$vc`" >nul 2>&1 & set PATH=$cmake;%PATH% & cd /d $BitNet & python setup_env.py -md $model -q i2_s -ld logs/bitnet_build"
cmd /c $cmdLine
$exe = Join-Path $BitNet "build\bin\Release\llama-cli.exe"
if (-not (Test-Path $exe)) { Write-Error "Build finished but missing $exe" }
Write-Host "OK: $exe"
