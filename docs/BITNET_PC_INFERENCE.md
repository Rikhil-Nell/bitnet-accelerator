# BitNet b1.58 PC inference (Microsoft stack)

Official runtime: **[microsoft/BitNet](https://github.com/microsoft/BitNet)** (`bitnet.cpp` / llama.cpp fork).

## Weights on this machine

**Installed:** `C:\AI\models\BitNet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf` (~1.19 GB) and tokenizer under `C:\AI\models\BitNet-b1.58-2B-4T\`.

Download (if missing):

```powershell
pip install huggingface_hub
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir C:\AI\models\BitNet-b1.58-2B-4T-gguf
```

Or full checkpoint (safetensors, for slice extract):

```powershell
huggingface-cli download microsoft/BitNet-b1.58-2B-4T --local-dir C:\AI\models\BitNet-b1.58-2B-4T
```

## Build bitnet.cpp (Windows)

Use **VS 2022 Developer PowerShell** (ClangCL + CMake). See upstream README.

```powershell
git clone --recursive https://github.com/microsoft/BitNet.git D:\src\BitNet
cd D:\src\BitNet
conda create -n bitnet-cpp python=3.10 -y
conda activate bitnet-cpp
pip install -r requirements.txt
make bitnet-cpp-build
```

Requires **VS 2022 ClangCL**; repo `setup_env.py` builds **`llama-cli` only** (skips llama UI server). Local fixes in `D:\src\BitNet` may be needed for I2_S link (`ggml-quants.c` + CPU CMake).

Inference (reference prompt):

```powershell
D:\src\BitNet\build\bin\Release\llama-cli.exe -m C:\AI\models\BitNet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf `
  -p "What is the capital of France?" -n 20 -t 8 -ngl 0
```

Evidence log: `build/bitnet/bitnet_cpp_france_nocnv.log` (mentions Paris).

## Capstone repo hooks

| Env | Purpose |
|-----|---------|
| `BITNET_CPP_ROOT` | Path to BitNet clone (default `D:/src/BitNet`) |
| `BITNET_MODEL_DIR` | HF model directory |
| `BITNET_GGUF` | Path to `ggml-model-i2_s.gguf` |

| Command | What |
|---------|------|
| `make test-bitnet-smoke` | Runs `run_inference.py` if clone + GGUF exist |
| `python tools/host/extract_bitlinear_slice.py` | 8×8 slice → JSON for FPGA kernel |
| `make test-bitnet-layer` | Kernel vs slice JSON |

**Encoding:** capstone RTL ≠ Microsoft I2_S block layout in GGUF. Use `extract_bitlinear_slice.py` (safetensors) for FPGA-aligned weights; use `i2s_word_to_capstone_word` only for nibble remap.
