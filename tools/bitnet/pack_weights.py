#!/usr/bin/env python3
"""Pack every BitNet ternary matrix into accelerator tile format for board DDR.

Tile = 8 output rows x 8 inputs = eight 16-bit weight words (row r of the tile
in word r; input j in bits [2j+1:2j]; codes 0=0, 1=+1, 2=-1, as in
tools/golden/ternary_gemv.py). Tiles are stored column-block major:
tile (cb, rg) at ((cb * N/8) + rg) * 16 bytes, matching the firmware loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_model import DEFAULT_GGUF, PROJ_NAMES, load_model  # noqa: E402

WEIGHT_BASE = 0x10000000
DEFAULT_OUT = Path("D:/xilinx-tools/bitnet_weights.bin")


def pack_tiles(trits: np.ndarray) -> np.ndarray:
    """(N, K) int8 trits -> uint16 words laid out [cb][rg][r]."""
    n, k = trits.shape
    if n % 8 or k % 8:
        raise ValueError(f"shape {trits.shape} not a multiple of 8")
    codes = np.zeros(trits.shape, dtype=np.uint16)
    codes[trits == 1] = 1
    codes[trits == -1] = 2
    c = codes.reshape(n // 8, 8, k // 8, 8)  # rg, r, cb, j
    shifts = (2 * np.arange(8, dtype=np.uint16))
    words = np.bitwise_or.reduce(c << shifts, axis=-1)  # rg, r, cb
    return np.ascontiguousarray(words.transpose(2, 0, 1)).reshape(-1)


def word_sum(buf: np.ndarray) -> int:
    """Checksum the firmware also computes: sum of little-endian u32 words mod 2^32."""
    return int(np.frombuffer(buf.tobytes(), dtype="<u4").sum(dtype=np.uint64) & 0xFFFFFFFF)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--with-norms",
        action="store_true",
        help="append RMS norm vectors for future on-board layer forward (changes bin layout)",
    )
    args = ap.parse_args()

    model = load_model(args.gguf)
    manifest = {"base": WEIGHT_BASE, "matrices": {}}
    off = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        for i, layer in enumerate(model.layers):
            for p in PROJ_NAMES:
                w = layer.proj[p]
                words = pack_tiles(w.trits)
                n, k = w.trits.shape
                f.write(words.astype("<u2").tobytes())
                manifest["matrices"][f"{i}.{p}"] = {
                    "addr": WEIGHT_BASE + off,
                    "bytes": words.nbytes,
                    "n": n,
                    "k": k,
                    "checksum": word_sum(words),
                }
                off += words.nbytes
            print(f"layer {i} packed ({off / 2**20:.0f} MiB)", flush=True)
        if args.with_norms:
            manifest["norms"] = {}
            for i, layer in enumerate(model.layers):
                for name, vec in (
                    ("attn_norm", layer.attn_norm),
                    ("attn_sub_norm", layer.attn_sub_norm),
                    ("ffn_norm", layer.ffn_norm),
                    ("ffn_sub_norm", layer.ffn_sub_norm),
                ):
                    raw = np.ascontiguousarray(vec.astype(np.float32))
                    f.write(raw.tobytes())
                    manifest["norms"][f"{i}.{name}"] = {
                        "addr": WEIGHT_BASE + off,
                        "bytes": raw.nbytes,
                        "n": int(raw.size),
                    }
                    off += raw.nbytes
            out_norm = np.ascontiguousarray(model.output_norm.astype(np.float32))
            f.write(out_norm.tobytes())
            manifest["output_norm"] = {
                "addr": WEIGHT_BASE + off,
                "bytes": out_norm.nbytes,
                "n": int(out_norm.size),
            }
            off += out_norm.nbytes
    manifest["total_bytes"] = off
    man_path = args.out.with_suffix(".json")
    man_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Wrote {args.out} ({off:,} bytes) and {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
