#!/usr/bin/env python3
"""Pack BitNet b1.58 2B-4T from the I2_S GGUF into the board image (board_image.py).

  python tools/bitnet/pack_board.py [--out D:/xilinx-tools/BITNET.BIN]

Copy the result to the root of a FAT32 SD card as BITNET.BIN.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board_image import (HDR, HEADER_BYTES, LAYER, MAGIC, SCALE_PROJ, VERSION,  # noqa: E402
                         ustride, word_sum)
from engine import quantize_rows_i8  # noqa: E402
from engine_format import ALIGN, NU, pack_t5, t5_k  # noqa: E402
from gguf_model import DEFAULT_GGUF, load_model  # noqa: E402

DEFAULT_OUT = Path("D:/xilinx-tools/BITNET.BIN")


class Writer:
    def __init__(self, f):
        self.f = f
        self.off = HEADER_BYTES
        self.sum = 0
        f.write(b"\0" * HEADER_BYTES)

    def put(self, arr: np.ndarray, align: int = ALIGN) -> int:
        pad = (-self.off) % align
        if pad:
            self.f.write(b"\0" * pad)
            self.off += pad
        data = np.ascontiguousarray(arr)
        b = data.tobytes()
        if len(b) % 4:
            b += b"\0" * (4 - len(b) % 4)
        start = self.off
        self.f.write(b)
        self.off += len(b)
        self.sum = (self.sum + word_sum(np.frombuffer(b, dtype=np.uint8))) & 0xFFFFFFFF
        return start

    def put_matrix(self, rows_t5: np.ndarray) -> tuple[int, int]:
        """(N, row_bytes) uint8 T5 rows -> NU unit slices at off + u * stride."""
        n, rb = rows_t5.shape
        if n % (2 * NU):
            raise ValueError(f"N={n} does not split into {NU} even slices")
        sl = n // NU * rb
        stride = ustride(sl)
        buf = np.zeros(stride * (NU - 1) + sl, dtype=np.uint8)
        for u in range(NU):
            buf[u * stride : u * stride + sl] = rows_t5[u * (n // NU) : (u + 1) * (n // NU)].reshape(-1)
        return self.put(buf), stride


def coarse_head(head_q: np.ndarray, head_scale: np.ndarray, chunk: int = 8192):
    """Ternary copy of the int8 head, exactly as the firmware ranks with it:
    keep |w| > 0.5 * mean|w| (2*E*|w| > sum|w|), scale = mean kept |w| * head_scale."""
    V, E = head_q.shape
    t = np.empty((V, E), dtype=np.int8)
    sc = np.empty(V, dtype=np.float32)
    for r0 in range(0, V, chunk):
        w = head_q[r0 : r0 + chunk].astype(np.int32)
        a = np.abs(w)
        keep = 2 * E * a > a.sum(axis=1, keepdims=True)
        tt = np.where(keep, np.sign(w), 0).astype(np.int8)
        t[r0 : r0 + chunk] = tt
        den = keep.sum(axis=1)
        num = (a * keep).sum(axis=1)
        sc[r0 : r0 + chunk] = np.where(den > 0, num.astype(np.float32) / np.maximum(den, 1), 0.0)             * head_scale[r0 : r0 + chunk]
    return t, sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    t0 = time.perf_counter()
    m = load_model(args.gguf)
    print(f"loaded model in {time.perf_counter() - t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    layer_rows = []
    strides: dict[str, int] = {}
    with args.out.open("wb") as f:
        w = Writer(f)
        for i, L in enumerate(m.layers):
            P = L.proj
            mats = {
                "qkv": np.concatenate([P["attn_q"].trits, P["attn_k"].trits, P["attn_v"].trits]),
                "o": P["attn_output"].trits,
                "gate_up": np.concatenate([P["ffn_gate"].trits, P["ffn_up"].trits]),
                "down": P["ffn_down"].trits,
            }
            offs = []
            for name, t in mats.items():
                off, stride = w.put_matrix(pack_t5(t))
                offs.append(off)
                strides[name] = stride
            layer_rows.append(offs)
            print(f"layer {i:2d}: {w.off / 2**20:6.0f} MiB", flush=True)

        head_q, head_scale = quantize_rows_i8(m.tok_embd)
        ct, cs = coarse_head(head_q, head_scale)
        coarse_off, coarse_stride = w.put_matrix(pack_t5(ct))
        del ct
        coarse_scale_off = w.put(cs.astype("<f4"))
        head_off = w.put(head_q)
        head_scale_off = w.put(head_scale.astype("<f4"))
        for i, L in enumerate(m.layers):
            vecs = [w.put(v.astype("<f4"), align=64)
                    for v in (L.attn_norm, L.attn_sub_norm, L.ffn_norm, L.ffn_sub_norm)]
            layer_rows[i] += vecs
        output_norm_off = w.put(m.output_norm.astype("<f4"), align=64)
        total = w.off

        table_off = HDR.size + (-HDR.size) % 16
        hdr = bytearray(HEADER_BYTES)
        HDR.pack_into(hdr, 0, MAGIC, VERSION, HEADER_BYTES, m.n_layer, m.n_embd, m.n_ff, m.n_head,
                      m.n_head_kv, head_q.shape[0], m.rope_theta, m.rms_eps, total, w.sum,
                      head_off, head_scale_off, output_norm_off, table_off,
                      strides["qkv"], strides["o"], strides["gate_up"], strides["down"],
                      coarse_off, coarse_scale_off, coarse_stride, t5_k(m.n_ff))
        for i, L in enumerate(m.layers):
            scales = [L.proj[p].scale for p in SCALE_PROJ]
            LAYER.pack_into(hdr, table_off + i * LAYER.size, *layer_rows[i], *scales)
        if table_off + m.n_layer * LAYER.size > HEADER_BYTES:
            raise SystemExit("layer table does not fit in the header")
        f.seek(0)
        f.write(hdr)

    print(f"wrote {args.out}: {total:,} bytes ({total / 2**20:.0f} MiB), checksum {w.sum:#010x}, "
          f"{time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
