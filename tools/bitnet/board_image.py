#!/usr/bin/env python3
"""The board weight image (BITNET.BIN on the SD card): layout, writer and reader.

The firmware copies the whole file to DDR at LOAD_ADDR and finds everything through
the 4 KiB header, so offsets never need to be compiled in.

Version 2 (engine v2, T5 = 1.6-bit ternary, engine_format.py):
  header (4096 B): struct bn_header2, then n_layer x struct bn_layer
  per layer, each matrix as 4 unit slices (rows split evenly), slice u at
  off + u * ustride, ustride = round_up(slice, 32 KiB) + 8 KiB so the four units
  never stream from the same DDR bank at once:
      qkv     [3840 x 2560]   rows q(2560) | k(640) | v(640)
      o       [2560 x 2560]
      gate_up [13824 x 2560]  rows gate(6912) | up(6912)
      down    [2560 x 6912]   K padded to 6920
  coarse head [vocab x 2560] T5: ternary copy of the int8 head (absmean threshold),
              coarse_scale [vocab] f32 = least-squares row scale * head_scale
  LM head     [vocab x 2560] int8, per-row absmax (embedding table, exact rescoring)
  head_scale  [vocab] f32
  per layer f32 vectors: attn_norm, attn_sub_norm, ffn_norm, ffn_sub_norm
  output_norm [2560] f32

All offsets are from the start of the file. The checksum is the u32 sum of all
little-endian words after the header. (Version 1, 2-bit ternary, is at git tag
board-v1.)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"BNZ2"
VERSION = 2
HEADER_BYTES = 4096
LOAD_ADDR = 0x0800_0000   # where the firmware places the image in DDR

HDR = struct.Struct("<4s8I2f6I8I")
HDR_FIELDS = ("magic", "version", "header_bytes", "n_layer", "n_embd", "n_ff", "n_head",
              "n_head_kv", "vocab", "rope_theta", "rms_eps", "total_bytes", "checksum",
              "head_off", "head_scale_off", "output_norm_off", "layer_table_off",
              "ustride_qkv", "ustride_o", "ustride_gate_up", "ustride_down",
              "coarse_off", "coarse_scale_off", "ustride_coarse", "k_pad_ff")
SLICE_ALIGN = 32 * 1024   # DDR bank period seen by the engine
SLICE_SKEW = 8 * 1024


def ustride(slice_bytes: int) -> int:
    return -(-slice_bytes // SLICE_ALIGN) * SLICE_ALIGN + SLICE_SKEW
LAYER = struct.Struct("<8I7f")
LAYER_FIELDS = ("qkv_off", "o_off", "gate_up_off", "down_off",
                "attn_norm_off", "attn_sub_norm_off", "ffn_norm_off", "ffn_sub_norm_off",
                "scale_q", "scale_k", "scale_v", "scale_o", "scale_gate", "scale_up", "scale_down")
SCALE_PROJ = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


@dataclass
class Header:
    magic: bytes
    version: int
    header_bytes: int
    n_layer: int
    n_embd: int
    n_ff: int
    n_head: int
    n_head_kv: int
    vocab: int
    rope_theta: float
    rms_eps: float
    total_bytes: int
    checksum: int
    head_off: int
    head_scale_off: int
    output_norm_off: int
    layer_table_off: int
    ustride_qkv: int
    ustride_o: int
    ustride_gate_up: int
    ustride_down: int
    coarse_off: int
    coarse_scale_off: int
    ustride_coarse: int
    k_pad_ff: int


def parse(buf: bytes | np.ndarray) -> tuple[Header, list[dict]]:
    raw = bytes(buf[:HEADER_BYTES])
    h = Header(*HDR.unpack_from(raw, 0))
    if h.magic != MAGIC or h.version != VERSION:
        raise ValueError(f"not a v{VERSION} board image: {h.magic!r} v{h.version}")
    layers = [dict(zip(LAYER_FIELDS, LAYER.unpack_from(raw, h.layer_table_off + i * LAYER.size)))
              for i in range(h.n_layer)]
    return h, layers


def word_sum(a: np.ndarray) -> int:
    return int(a.view("<u4").sum(dtype=np.uint64) & 0xFFFFFFFF)
