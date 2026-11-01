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

The reference reply to the France prompt mentions Paris.

## Local patches

The fixes needed to build on this machine (I2_S quantize/dequantize in
`ggml-quants.c`, a CPU-backend guard in the LUT transform, `llama-cli`-only build)
are saved in `tools/setup/bitnet-cpp-patches/`, with the base commits in
`BASE_COMMITS.txt`. To re-apply them to a fresh clone:

```powershell
cd D:\src\BitNet; git apply <repo>\tools\setup\bitnet-cpp-patches\bitnet.patch
cd 3rdparty\llama.cpp; git apply <repo>\tools\setup\bitnet-cpp-patches\llama.cpp.patch
```

## Repo hooks

| Env | Purpose |
|-----|---------|
| `BITNET_CPP_ROOT` | BitNet clone (default `D:/src/BitNet`); also supplies `gguf-py` |
| `BITNET_HF_DIR` | HF model dir with `tokenizer.json` |
| `BITNET_GGUF` | Path to `ggml-model-i2_s.gguf` |

`make generate-emulated PROMPT="..."` runs the NumPy reference model
(`tools/bitnet/engine.py`). It matches bitnet.cpp's I2_S unpacking and int8
activation quantization, and answers "The capital of France is Paris."
