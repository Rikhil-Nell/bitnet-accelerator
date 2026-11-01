#!/usr/bin/env python3
"""Run the reference forward pass from the packed board image (BITNET.BIN) alone.

This validates the packer end to end: the image is read back the way the firmware
reads it (header table, engine ternary format, int8 head) and must reproduce the
int8-head reference answer.

  python tools/bitnet/board_emu.py -p "What is the capital of France?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board_image import parse  # noqa: E402
from engine import BitNetEngine, EmulatedAccelerator  # noqa: E402
from engine_format import unpack_t5  # noqa: E402
from generate import HF_DIR, STOP_IDS, chat_prompt  # noqa: E402
from gguf_model import BitNetModel, Layer, TernaryMatrix  # noqa: E402
from pack_board import DEFAULT_OUT  # noqa: E402


def load_image(path: Path, n_layer_limit: int | None = None) -> tuple[BitNetModel, np.ndarray, np.ndarray]:
    img = np.memmap(path, dtype=np.uint8, mode="r")
    h, table = parse(img)
    E, F, V = h.n_embd, h.n_ff, h.vocab
    kv = E // h.n_head * h.n_head_kv

    def f32(off: int, n: int) -> np.ndarray:
        return np.frombuffer(img[off : off + 4 * n].tobytes(), dtype="<f4").copy()

    def tern(off: int, n: int, k: int, stride: int) -> np.ndarray:
        """v2: T5 rows, 4 unit slices at off + u * stride"""
        rb = (k + 39) // 40 * 8
        r = n // 4
        b = np.concatenate([np.asarray(img[off + u * stride : off + u * stride + r * rb]) for u in range(4)])
        return unpack_t5(b.reshape(n, rb), k)

    layers = []
    for i, t in enumerate(table[: n_layer_limit or h.n_layer]):
        qkv = tern(t["qkv_off"], E + 2 * kv, E, h.ustride_qkv)
        gu = tern(t["gate_up_off"], 2 * F, E, h.ustride_gate_up)
        L = Layer(f32(t["attn_norm_off"], E), f32(t["attn_sub_norm_off"], E),
                  f32(t["ffn_norm_off"], E), f32(t["ffn_sub_norm_off"], F))
        parts = {
            "attn_q": qkv[:E], "attn_k": qkv[E : E + kv], "attn_v": qkv[E + kv :],
            "attn_output": tern(t["o_off"], E, E, h.ustride_o),
            "ffn_gate": gu[:F], "ffn_up": gu[F:],
            "ffn_down": tern(t["down_off"], E, F, h.ustride_down),
        }
        scales = [t[s] for s in ("scale_q", "scale_k", "scale_v", "scale_o", "scale_gate", "scale_up", "scale_down")]
        for (name, trits), sc in zip(parts.items(), scales):
            L.proj[name] = TernaryMatrix(name, trits, float(sc))
        layers.append(L)

    head_q = np.frombuffer(img[h.head_off : h.head_off + V * E].tobytes(), dtype=np.int8).reshape(V, E)
    head_scale = f32(h.head_scale_off, V)
    model = BitNetModel(n_layer=len(layers), n_embd=E, n_ff=F, n_head=h.n_head, n_head_kv=h.n_head_kv,
                        rope_theta=h.rope_theta, rms_eps=h.rms_eps, tok_embd=np.zeros((1, E), np.float32),
                        output_norm=f32(h.output_norm_off, E), layers=layers)
    return model, head_q, head_scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=DEFAULT_OUT)
    ap.add_argument("-p", "--prompt", default="What is the capital of France?")
    ap.add_argument("-n", "--max-tokens", type=int, default=16)
    ap.add_argument("--verify-checksum", action="store_true")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))

    t0 = time.perf_counter()
    if args.verify_checksum:
        from board_image import HEADER_BYTES, word_sum
        img = np.memmap(args.image, dtype=np.uint8, mode="r")
        h, _ = parse(img)
        got = word_sum(np.asarray(img[HEADER_BYTES : h.total_bytes]))
        print(f"checksum {got:#010x} vs header {h.checksum:#010x}: {'OK' if got == h.checksum else 'BAD'}")
    model, head_q, head_scale = load_image(args.image)
    eng = BitNetEngine(model, EmulatedAccelerator(model))
    eng.head_int8, eng.head_q, eng.head_scale = True, head_q, head_scale
    print(f"[load] {time.perf_counter() - t0:.0f}s", flush=True)

    ids = tok.encode(chat_prompt(args.prompt), add_special_tokens=False).ids
    logits = eng.forward(ids)
    out = []
    for _ in range(args.max_tokens):
        nxt = int(np.argmax(logits))
        if nxt in STOP_IDS:
            break
        out.append(nxt)
        logits = eng.forward([nxt])
    print(f"prompt ids: {ids}")
    print(f"output ids: {out}")
    print(f"output: {tok.decode(out)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
