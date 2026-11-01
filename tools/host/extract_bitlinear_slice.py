#!/usr/bin/env python3
"""Extract 8x8 BitLinear slice from HF BitNet weights -> capstone JSON vector."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "golden"))
sys.path.insert(0, str(ROOT / "tools" / "host"))

from bitlinear_kernel import bitlinear_gemv_int8
from bitnet_paths import discover_model_dir
from ternary_gemv import pack_weight_word


def weight_to_trit(w: float) -> int:
    wi = int(round(float(w)))
    return max(-1, min(1, wi))


def load_tensor(repo_id: str, tensor: str, model_dir: Path | None):
    try:
        import torch
        from safetensors import safe_open
    except ImportError as e:
        raise SystemExit("pip install torch safetensors huggingface_hub") from e

    if model_dir and (model_dir / "model.safetensors.index.json").exists():
        local = model_dir
    else:
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(repo_id, local_dir=model_dir))

    index_path = local / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard = index["weight_map"].get(tensor)
        if not shard:
            raise SystemExit(f"tensor not in index: {tensor}")
        st_path = local / shard
    else:
        st_path = local / "model.safetensors"
        if not st_path.exists():
            shards = list(local.glob("*.safetensors"))
            if len(shards) == 1:
                st_path = shards[0]
            else:
                raise SystemExit(f"No safetensors in {local}")

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        if tensor not in f.keys():
            raise SystemExit(f"{tensor} not in {st_path}")
        return f.get_tensor(tensor)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="microsoft/BitNet-b1.58-2B-4T")
    ap.add_argument("--tensor", default="model.layers.0.self_attn.q_proj.weight")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "tools/golden/vectors/bitnet_layer0_q_8x8.json",
    )
    ap.add_argument("--model-dir", type=Path, default=None)
    args = ap.parse_args()

    model_dir = args.model_dir or discover_model_dir()
    t = load_tensor(args.repo_id, args.tensor, model_dir)
    if t.dim() != 2:
        raise SystemExit(f"expected 2D weight, got shape {tuple(t.shape)}")

    r, c = args.rows, args.cols
    if t.shape[0] < r or t.shape[1] < c:
        raise SystemExit(f"tensor too small {tuple(t.shape)} for {r}x{c}")

    slice_ = t[:r, :c].float()
    trits = [[weight_to_trit(slice_[i, j].item()) for j in range(c)] for i in range(r)]
    words = [pack_weight_word(row) for row in trits]

    import random

    random.seed(0)
    acts = [random.randint(-7, 7) for _ in range(c)]
    expected = bitlinear_gemv_int8(acts, words)

    out = {
        "name": "bitnet_hf_slice",
        "source": {"repo_id": args.repo_id, "tensor": args.tensor, "slice": [r, c]},
        "encoding": "capstone_rtl",
        "activations": acts,
        "weight_words_hex": [f"{w:04x}" for w in words],
        "expected_outputs": expected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
