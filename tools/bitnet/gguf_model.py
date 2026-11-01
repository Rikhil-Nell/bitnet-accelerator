#!/usr/bin/env python3
"""Load Microsoft BitNet b1.58 2B-4T from its I2_S GGUF into NumPy arrays.

I2_S layout (bitnet.cpp, x86 ACT_PARALLEL, QK_I2_S=128): the row-major ternary
matrix is cut into blocks of 128 weights stored in 32 bytes; byte p of a block
holds weights p, p+32, p+64, p+96 in bits [7:6], [5:4], [3:2], [1:0].
Codes 0/1/2 mean -1/0/+1. One float32 per-tensor scale follows the n/4 bytes.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BITNET_ROOT = Path(os.environ.get("BITNET_CPP_ROOT", "D:/src/BitNet"))
sys.path.insert(0, str(BITNET_ROOT / "3rdparty" / "llama.cpp" / "gguf-py"))

import gguf  # noqa: E402

DEFAULT_GGUF = Path(
    os.environ.get(
        "BITNET_GGUF",
        "C:/AI/models/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf",
    )
)

PROJ_NAMES = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


@dataclass
class TernaryMatrix:
    """W[n_out, n_in] in {-1, 0, +1} plus the per-tensor scale."""

    name: str
    trits: np.ndarray  # int8, shape (n_out, n_in)
    scale: float


@dataclass
class Layer:
    attn_norm: np.ndarray
    attn_sub_norm: np.ndarray
    ffn_norm: np.ndarray
    ffn_sub_norm: np.ndarray
    proj: dict[str, TernaryMatrix] = field(default_factory=dict)


@dataclass
class BitNetModel:
    n_layer: int
    n_embd: int
    n_ff: int
    n_head: int
    n_head_kv: int
    rope_theta: float
    rms_eps: float
    tok_embd: np.ndarray  # float32 (vocab, n_embd); also the tied LM head
    output_norm: np.ndarray
    layers: list[Layer]


def unpack_i2s(raw: np.ndarray, n_out: int, n_in: int) -> np.ndarray:
    """Packed I2_S bytes -> int8 trits of shape (n_out, n_in)."""
    n = n_out * n_in
    if n % 128 or raw.size != n // 4:
        raise ValueError(f"bad I2_S size: {raw.size} bytes for {n_out}x{n_in}")
    blocks = raw.reshape(-1, 32)
    groups = np.stack([(blocks >> s) & 0x3 for s in (6, 4, 2, 0)], axis=1)  # (nb, 4, 32)
    codes = groups.reshape(-1)
    if np.any(codes == 3):
        raise ValueError("I2_S code 3 found; layout assumption is wrong")
    return (codes.astype(np.int8) - 1).reshape(n_out, n_in)


def _field(reader: gguf.GGUFReader, key: str):
    return reader.fields[key].contents()


def load_model(path: Path = DEFAULT_GGUF, n_layer_limit: int | None = None) -> BitNetModel:
    reader = gguf.GGUFReader(str(path))
    arch = _field(reader, "general.architecture")
    tensors = {t.name: t for t in reader.tensors}
    mm = np.memmap(path, dtype=np.uint8, mode="r")

    def f32(name: str) -> np.ndarray:
        return np.array(tensors[name].data, dtype=np.float32)

    def ternary(name: str) -> TernaryMatrix:
        t = tensors[name]
        n_in, n_out = (int(x) for x in t.shape)
        nbytes = n_out * n_in // 4
        off = int(t.data_offset)
        raw = np.asarray(mm[off : off + nbytes])
        scale = float(np.frombuffer(mm[off + nbytes : off + nbytes + 4].tobytes(), dtype=np.float32)[0])
        return TernaryMatrix(name, unpack_i2s(raw, n_out, n_in), scale)

    n_layer = int(_field(reader, f"{arch}.block_count"))
    if n_layer_limit is not None:
        n_layer = min(n_layer, n_layer_limit)

    layers = []
    for i in range(n_layer):
        p = f"blk.{i}."
        layer = Layer(
            attn_norm=f32(p + "attn_norm.weight"),
            attn_sub_norm=f32(p + "attn_sub_norm.weight"),
            ffn_norm=f32(p + "ffn_norm.weight"),
            ffn_sub_norm=f32(p + "ffn_sub_norm.weight"),
        )
        for proj in PROJ_NAMES:
            layer.proj[proj] = ternary(f"{p}{proj}.weight")
        layers.append(layer)

    tok = tensors["token_embd.weight"]
    n_embd, vocab = (int(x) for x in tok.shape)
    tok_embd = np.array(tok.data, dtype=np.float32).reshape(vocab, n_embd)

    return BitNetModel(
        n_layer=n_layer,
        n_embd=n_embd,
        n_ff=int(_field(reader, f"{arch}.feed_forward_length")),
        n_head=int(_field(reader, f"{arch}.attention.head_count")),
        n_head_kv=int(_field(reader, f"{arch}.attention.head_count_kv")),
        rope_theta=float(_field(reader, f"{arch}.rope.freq_base")),
        rms_eps=float(_field(reader, f"{arch}.attention.layer_norm_rms_epsilon")),
        tok_embd=tok_embd,
        output_norm=f32("output_norm.weight"),
        layers=layers,
    )


if __name__ == "__main__":
    m = load_model(n_layer_limit=1)
    for name, w in m.layers[0].proj.items():
        vals, counts = np.unique(w.trits, return_counts=True)
        frac = {int(v): round(c / w.trits.size, 3) for v, c in zip(vals, counts)}
        print(f"{name:12s} {w.trits.shape} scale={w.scale:.4f} trits={frac}")
    print("tok_embd", m.tok_embd.shape, "n_head", m.n_head, "n_head_kv", m.n_head_kv)
