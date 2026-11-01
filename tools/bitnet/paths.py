"""Machine-specific locations, overridable by environment variables (see SETUP.md).

BITNET_WORK      work directory for large generated files (board image, Vitis workspace,
                 corpora); default D:/xilinx-tools
BITNET_GGUF      bitnet.cpp GGUF of BitNet b1.58 2B-4T (gguf_model.py)
BITNET_HF_DIR    Hugging Face model directory with tokenizer.json (generate.py)
BITNET_CPP_ROOT  bitnet.cpp checkout, for its gguf-py package (gguf_model.py)
"""
import os
from pathlib import Path

WORK = Path(os.environ.get("BITNET_WORK", "D:/xilinx-tools"))
