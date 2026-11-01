#!/usr/bin/env python3
"""The board model (BITNET.BIN: ternary layers, int8 activations, int8 LM head) on a
CUDA GPU, batched. Same math as engine.py / the firmware: every BitLinear is an exact
int8 x ternary integer matmul (torch._int_mm), everything else float32.

Used to generate many greedy answers quickly (speculative-decoding statistics, traces
for the write-up). Not bit-identical to the board: float32 sums are re-associated, so
outputs can differ at near-ties, exactly like engine.py vs the board.

  python tools/bitnet/gpu_model.py -p "What is the capital of France?" -p "Who painted the Mona Lisa?"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board_image import parse  # noqa: E402
from pack_board import DEFAULT_OUT  # noqa: E402

STOP_IDS = (128001, 128009)


def chat_prompt(q: str) -> str:
    return f"<|begin_of_text|>User: {q.strip()}<|eot_id|>Assistant: "


class GpuBitNet:
    def __init__(self, image: Path = DEFAULT_OUT, device: str = "cuda"):
        self.dev = torch.device(device)
        img = np.memmap(image, dtype=np.uint8, mode="r")
        h, table = parse(img)
        self.h = h
        E, F, V = h.n_embd, h.n_ff, h.vocab
        self.E, self.F, self.V = E, F, V
        self.nh, self.nkv = h.n_head, h.n_head_kv
        self.hd = E // self.nh
        self.kv = self.hd * self.nkv
        self.eps = float(h.rms_eps)

        def f32(off: int, n: int) -> torch.Tensor:
            return torch.from_numpy(np.frombuffer(img[off : off + 4 * n].tobytes(), dtype="<f4").copy()).to(self.dev)

        def tern(off: int, n: int, k: int, stride: int) -> torch.Tensor:
            """v2 image: T5 rows (t5_k(k)/5 bytes), 4 unit slices at off + u * stride"""
            rb = (k + 39) // 40 * 8
            r = n // 4
            parts = [np.asarray(img[off + u * stride : off + u * stride + r * rb]) for u in range(4)]
            b = torch.from_numpy(np.concatenate(parts)).to(self.dev).view(n, rb).to(torch.int16)
            d = []
            for _ in range(5):
                d.append((b % 3 - 1).to(torch.int8))
                b = torch.div(b, 3, rounding_mode="floor")
            t = torch.stack(d, -1).view(n, rb * 5)[:, :k].contiguous()
            return t.t()   # (k, n): column-major view of row-major W, what _int_mm wants

        self.layers = []
        for t in table:
            self.layers.append({
                "qkv": tern(t["qkv_off"], E + 2 * self.kv, E, h.ustride_qkv),
                "o": tern(t["o_off"], E, E, h.ustride_o),
                "gu": tern(t["gate_up_off"], 2 * F, E, h.ustride_gate_up),
                "down": tern(t["down_off"], E, F, h.ustride_down),
                "attn_norm": f32(t["attn_norm_off"], E), "attn_sub_norm": f32(t["attn_sub_norm_off"], E),
                "ffn_norm": f32(t["ffn_norm_off"], E), "ffn_sub_norm": f32(t["ffn_sub_norm_off"], F),
                "s": {k: float(t["scale_" + k]) for k in ("q", "k", "v", "o", "gate", "up", "down")},
            })
        self.head_q = torch.from_numpy(np.asarray(img[h.head_off : h.head_off + V * E])).to(self.dev).view(torch.int8).view(V, E)
        self.head_scale = f32(h.head_scale_off, V)
        self.out_norm = f32(h.output_norm_off, E)
        half = self.hd // 2
        self.inv_freq = 1.0 / (float(h.rope_theta) ** (torch.arange(half, dtype=torch.float64) * 2 / self.hd))

    # ---- pieces (all float32 unless noted) ----
    def rms(self, x, w):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps) * w

    @staticmethod
    def quant(x):
        amax = x.abs().amax(-1, keepdim=True)
        s = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
        v = x * s
        q = torch.sign(v) * torch.floor(v.abs() + 0.5)
        return q.clamp(-128, 127).to(torch.int8), s

    @staticmethod
    def imm(a8, w8):
        """exact int32 a8 (M,K) @ w8 (K,N); _int_mm needs M > 16"""
        M = a8.shape[0]
        if M <= 16:
            a8 = torch.cat([a8, a8.new_zeros(17 - M, a8.shape[1])])
        return torch._int_mm(a8, w8)[:M]

    def bitlinear(self, x, w8):
        q, s = self.quant(x)
        return self.imm(q, w8).float() / s, s

    def rope(self, x, pos):
        """x (B, T, H, D), pos (B, T) int64"""
        ang = pos.double().cpu()[..., None] * self.inv_freq
        cos = torch.cos(ang).float().to(self.dev)[:, :, None, :]
        sin = torch.sin(ang).float().to(self.dev)[:, :, None, :]
        h = self.hd // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    @torch.inference_mode()
    def generate(self, prompts: list[list[int]], max_new: int = 64, trace=None) -> list[list[int]]:
        """Greedy, batched, left-padded. trace(step, info) gets per-step tensors if given."""
        B = len(prompts)
        P = max(len(p) for p in prompts)
        S = P + max_new
        pad = torch.tensor([P - len(p) for p in prompts], device=self.dev)
        ids = torch.zeros(B, P, dtype=torch.long, device=self.dev)
        for b, p in enumerate(prompts):
            ids[b, P - len(p):] = torch.tensor(p, device=self.dev)
        K = torch.zeros(len(self.layers), B, self.nkv, S, self.hd, device=self.dev)
        Vc = torch.zeros_like(K)
        col = torch.arange(S, device=self.dev)
        out = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        cur, start = ids, 0
        for step in range(max_new):
            T = cur.shape[1]
            cols = torch.arange(start, start + T, device=self.dev)
            pos = (cols[None, :] - pad[:, None]).clamp(min=0)
            x = self.head_q[cur].float() * self.head_scale[cur][..., None]          # (B, T, E)
            # key j visible to query i of sequence b: pad_b <= j <= i
            # (padding rows see only themselves, so they stay finite: 0 * NaN would leak)
            kc = col[None, None, :start + T]
            mask = ((kc >= pad[:, None, None]) & (kc <= cols[None, :, None])) | (kc == cols[None, :, None])
            for li, L in enumerate(self.layers):
                h = self.rms(x, L["attn_norm"]).view(B * T, -1)
                y, _ = self.bitlinear(h, L["qkv"])
                s = L["s"]
                q = (y[:, :self.E] * s["q"]).view(B, T, self.nh, self.hd)
                k = (y[:, self.E:self.E + self.kv] * s["k"]).view(B, T, self.nkv, self.hd)
                v = (y[:, self.E + self.kv:] * s["v"]).view(B, T, self.nkv, self.hd)
                q, k = self.rope(q, pos), self.rope(k, pos)
                K[li, :, :, start:start + T] = k.transpose(1, 2)
                Vc[li, :, :, start:start + T] = v.transpose(1, 2)
                kk = K[li, :, :, :start + T].repeat_interleave(self.nh // self.nkv, dim=1)   # (B, nh, S', D)
                vv = Vc[li, :, :, :start + T].repeat_interleave(self.nh // self.nkv, dim=1)
                att = (q.transpose(1, 2) @ kk.transpose(-1, -2)) / (self.hd ** 0.5)       # (B, nh, T, S')
                att = att.masked_fill(~mask[:, None], float("-inf")).softmax(-1)
                o = (att @ vv).transpose(1, 2).reshape(B * T, self.E)
                o = self.rms(o, L["attn_sub_norm"])
                y, _ = self.bitlinear(o, L["o"])
                x = x + (y * s["o"]).view(B, T, self.E)
                h = self.rms(x, L["ffn_norm"]).view(B * T, -1)
                y, _ = self.bitlinear(h, L["gu"])
                g, u = y[:, :self.F] * s["gate"], y[:, self.F:] * s["up"]
                a = self.rms(torch.relu(g).square() * u, L["ffn_sub_norm"])
                y, _ = self.bitlinear(a, L["down"])
                x = x + (y * s["down"]).view(B, T, self.E)
            hl = self.rms(x[:, -1], self.out_norm)
            hq, _ = self.quant(hl)
            logits = self.imm(hq, self.head_q.t()).float() * self.head_scale
            nxt = logits.argmax(-1)
            if trace is not None:
                trace(step, {"logits": logits, "hq": hq})
            for b in range(B):
                if not done[b]:
                    out[b].append(int(nxt[b]))
            done |= (nxt == STOP_IDS[0]) | (nxt == STOP_IDS[1])
            if bool(done.all()):
                break
            start += T
            cur = nxt[:, None]
        return [[t for t in o if t not in STOP_IDS] for o in out]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt", action="append", default=[])
    ap.add_argument("-n", type=int, default=48)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    from generate import HF_DIR
    tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    t0 = time.perf_counter()
    m = GpuBitNet()
    print(f"[load] {time.perf_counter() - t0:.1f}s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB on GPU")
    prompts = args.prompt or ["What is the capital of France?"]
    t0 = time.perf_counter()
    outs = m.generate([tok.encode(chat_prompt(p), add_special_tokens=False).ids for p in prompts], args.n)
    dt = time.perf_counter() - t0
    for p, o in zip(prompts, outs):
        print(f"{p!r} -> {tok.decode(o)!r}")
    print(f"[gen] {sum(map(len, outs))} tokens in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
