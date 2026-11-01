#!/usr/bin/env python3
"""BitNet b1.58 2B-4T forward pass with every BitLinear routed through a backend.

The backend only sees what the FPGA accelerator sees: INT8 activations and
ternary weights, returning exact integer dot products. Everything else
(RMSNorm, RoPE, attention, ReLU^2, LM head) runs on the host in float32,
following bitnet.cpp / HF BitNetForCausalLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from gguf_model import PROJ_NAMES, BitNetModel


def quantize_act_i8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row absmax INT8, identical to bitnet.cpp quantize_row_i8_s (roundf = half away from zero)."""
    amax = np.max(np.abs(x), axis=-1, keepdims=True).astype(np.float32)
    s = np.where(amax > 0, np.float32(127.0) / amax, np.float32(0.0)).astype(np.float32)
    v = x * s
    q = np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))
    return np.clip(q, -128, 127).astype(np.int8), s


class EmulatedAccelerator:
    """Host model of the RTL: y[t, n] = sum_k W[n, k] * xq[t, k] as exact integers.

    float32 BLAS is exact here: every partial sum is an integer below 2^24
    (|sum| <= 6912 * 128).
    """

    name = "emulated"

    def __init__(self, model: BitNetModel):
        self.w = {}
        for i, layer in enumerate(model.layers):
            for p in PROJ_NAMES:
                self.w[(i, p)] = layer.proj[p].trits.astype(np.float32)
        self.calls = 0
        self.macs = 0

    def matmul(self, layer: int, proj: str, xq: np.ndarray) -> np.ndarray:
        w = self.w[(layer, proj)]
        self.calls += 1
        self.macs += xq.shape[0] * w.size
        return (xq.astype(np.float32) @ w.T).astype(np.int32)


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)) * w


@dataclass
class KVCache:
    k: list = field(default_factory=list)
    v: list = field(default_factory=list)


class BitNetEngine:
    def __init__(self, model: BitNetModel, backend):
        self.m = model
        self.backend = backend
        self.head_dim = model.n_embd // model.n_head
        half = self.head_dim // 2
        self.inv_freq = (1.0 / (model.rope_theta ** (np.arange(half, dtype=np.float64) * 2 / self.head_dim))).astype(
            np.float64
        )
        self.cache = [KVCache() for _ in range(model.n_layer)]
        self.n_past = 0
        self.bitlinear_time = 0.0

    def reset(self) -> None:
        self.cache = [KVCache() for _ in range(self.m.n_layer)]
        self.n_past = 0

    def bitlinear(self, layer: int, proj: str, x: np.ndarray) -> np.ndarray:
        xq, s = quantize_act_i8(x)
        t0 = time.perf_counter()
        y = self.backend.matmul(layer, proj, xq)
        self.bitlinear_time += time.perf_counter() - t0
        w_scale = np.float32(self.m.layers[layer].proj[proj].scale)
        return (y.astype(np.float32) / s) * w_scale

    def bitlinear_qkv(
        self, layer: int, x: np.ndarray, nh: int, nkv: int, hd: int, t: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xq, s = quantize_act_i8(x)
        t0 = time.perf_counter()
        fuse = getattr(self.backend, "matmul_fused", None)
        if fuse is not None:
            yq, yk, yv = fuse(layer, ("attn_q", "attn_k", "attn_v"), xq)
        else:
            yq = self.backend.matmul(layer, "attn_q", xq)
            yk = self.backend.matmul(layer, "attn_k", xq)
            yv = self.backend.matmul(layer, "attn_v", xq)
        self.bitlinear_time += time.perf_counter() - t0
        L = self.m.layers[layer]
        sq = np.float32(L.proj["attn_q"].scale)
        sk = np.float32(L.proj["attn_k"].scale)
        sv = np.float32(L.proj["attn_v"].scale)
        q = ((yq.astype(np.float32) / s) * sq).reshape(t, nh, hd)
        k = ((yk.astype(np.float32) / s) * sk).reshape(t, nkv, hd)
        v = ((yv.astype(np.float32) / s) * sv).reshape(t, nkv, hd)
        return q, k, v

    def bitlinear_ffn_gate_up(self, layer: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fused gate+up when the backend supports it (one PL pass over K tiles)."""
        xq, s = quantize_act_i8(x)
        t0 = time.perf_counter()
        fuse = getattr(self.backend, "matmul_fused", None)
        if fuse is not None and getattr(self.backend, "fuse_ffn", False):
            yg, yu = fuse(layer, ("ffn_gate", "ffn_up"), xq)
        else:
            yg = self.backend.matmul(layer, "ffn_gate", xq)
            yu = self.backend.matmul(layer, "ffn_up", xq)
        self.bitlinear_time += time.perf_counter() - t0
        L = self.m.layers[layer]
        sg, su = np.float32(L.proj["ffn_gate"].scale), np.float32(L.proj["ffn_up"].scale)
        g = (yg.astype(np.float32) / s) * sg
        u = (yu.astype(np.float32) / s) * su
        return g, u

    def rope(self, x: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """NEOX-style rotary embedding (rotate halves), x: (T, H, D)."""
        ang = pos[:, None].astype(np.float64) * self.inv_freq[None, :]
        cos = np.cos(ang).astype(np.float32)[:, None, :]
        sin = np.sin(ang).astype(np.float32)[:, None, :]
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    def forward(self, tokens: list[int]) -> np.ndarray:
        """Run tokens through the model (appending to the KV cache); return last-token logits."""
        m = self.m
        T = len(tokens)
        pos = np.arange(self.n_past, self.n_past + T)
        hd, nh, nkv = self.head_dim, m.n_head, m.n_head_kv
        group = nh // nkv
        x = m.tok_embd[np.array(tokens)].astype(np.float32)

        verbose = getattr(self.backend, "verbose", False)
        for i, L in enumerate(m.layers):
            if verbose:
                print(f"[layer {i}/{m.n_layer}] T={T} bitlinear...", flush=True)
            h = rms_norm(x, L.attn_norm, m.rms_eps)
            q, k, v = self.bitlinear_qkv(i, h, nh, nkv, hd, T)
            q = self.rope(q, pos)
            k = self.rope(k, pos)

            c = self.cache[i]
            c.k.append(k)
            c.v.append(v)
            K = np.concatenate(c.k, axis=0)  # (S, nkv, hd)
            V = np.concatenate(c.v, axis=0)
            S = K.shape[0]

            Kh = np.repeat(K, group, axis=1).transpose(1, 0, 2)  # (nh, S, hd)
            Vh = np.repeat(V, group, axis=1).transpose(1, 0, 2)
            Qh = q.transpose(1, 0, 2)  # (nh, T, hd)
            att = (Qh @ Kh.transpose(0, 2, 1)) / np.float32(np.sqrt(hd))
            causal = (np.arange(S)[None, :] > (self.n_past + np.arange(T))[:, None])
            att = np.where(causal[None], np.float32(-np.inf), att)
            att = att - att.max(axis=-1, keepdims=True)
            att = np.exp(att)
            att /= att.sum(axis=-1, keepdims=True)
            o = (att @ Vh).transpose(1, 0, 2).reshape(T, nh * hd)

            o = rms_norm(o, L.attn_sub_norm, m.rms_eps)
            x = x + self.bitlinear(i, "attn_output", o)

            h = rms_norm(x, L.ffn_norm, m.rms_eps)
            g, u = self.bitlinear_ffn_gate_up(i, h)
            a = np.square(np.maximum(g, 0)) * u
            a = rms_norm(a, L.ffn_sub_norm, m.rms_eps)
            x = x + self.bitlinear(i, "ffn_down", a)

        self.n_past += T
        h = rms_norm(x[-1:], m.output_norm, m.rms_eps)
        return (h @ m.tok_embd.T)[0]
