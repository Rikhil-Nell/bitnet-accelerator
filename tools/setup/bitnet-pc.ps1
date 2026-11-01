# BitNet.cpp PC inference helper (run from repo root).
# Full steps: docs/BITNET_PC_INFERENCE.md

$BitNetRoot = if ($env:BITNET_CPP_ROOT) { $env:BITNET_CPP_ROOT } else { "D:\src\BitNet" }
$ModelGguf = "C:\AI\models\BitNet-b1.58-2B-4T-gguf"
$ModelHf = "C:\AI\models\BitNet-b1.58-2B-4T"

Write-Host "BITNET_CPP_ROOT -> $BitNetRoot"
Write-Host "Suggested GGUF dir -> $ModelGguf"
Write-Host ""
Write-Host "Clone (VS dev shell recommended for build):"
Write-Host "  git clone --recursive https://github.com/microsoft/BitNet.git $BitNetRoot"
Write-Host ""
Write-Host "Download weights:"
Write-Host "  huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir $ModelGguf"
Write-Host ""
Write-Host "Reference generation (NumPy):"
Write-Host "  make generate-emulated PROMPT=`"What is the capital of France?`""
