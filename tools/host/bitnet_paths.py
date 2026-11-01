"""Resolve BitNet.cpp model paths from env and common local layouts."""
from __future__ import annotations

import os
from pathlib import Path


def discover_model_dir() -> Path | None:
    for key in ("BITNET_MODEL_DIR", "BITNET_GGUF"):
        raw = os.environ.get(key)
        if raw:
            p = Path(raw)
            if p.is_file():
                return p.parent
            if p.is_dir():
                return p

    candidates = [
        Path(os.environ.get("BITNET_CPP_ROOT", "D:/src/BitNet")) / "models" / "BitNet-b1.58-2B-4T",
        Path("D:/src/BitNet/models/BitNet-b1.58-2B-4T"),
        Path("C:/AI/models/BitNet-b1.58-2B-4T"),
        Path("C:/AI/BitNet-b1.58-2B-4T"),
        Path("C:/AI/models/BitNet-b1.58-2B-4T-gguf"),
        Path("C:/AI/BitNet-b1.58-2B-4T-gguf"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for base in candidates:
        if not base.exists():
            continue
        if base.name == "hub":
            for child in base.glob("models--microsoft--BitNet*"):
                snap = child / "snapshots"
                if snap.is_dir():
                    for s in snap.iterdir():
                        if s.is_dir():
                            return s
            continue
        return base
    return None


def discover_gguf() -> Path | None:
    env = os.environ.get("BITNET_GGUF")
    if env:
        p = Path(env)
        if p.is_file():
            return p

    model_dir = discover_model_dir()
    if model_dir:
        for name in ("ggml-model-i2_s.gguf", "ggml-model-i2_s-gguf", "model.gguf"):
            hit = model_dir / name
            if hit.is_file():
                return hit
        for hit in model_dir.rglob("*.gguf"):
            if "i2_s" in hit.name.lower() or "bitnet" in hit.name.lower():
                return hit
        any_gguf = next(model_dir.rglob("*.gguf"), None)
        if any_gguf:
            return any_gguf
    return None


def discover_bitnet_cpp_root() -> Path | None:
    raw = os.environ.get("BITNET_CPP_ROOT", "D:/src/BitNet")
    p = Path(raw)
    if (p / "run_inference.py").is_file():
        return p
    return None
