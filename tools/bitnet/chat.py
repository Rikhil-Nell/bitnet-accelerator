#!/usr/bin/env python3
"""Chat with BitNet running on the Zybo (fpga/ps/bitnet_zybo) over UART.

  python tools/bitnet/chat.py                       # interactive, new context per question
  python tools/bitnet/chat.py --multi               # keep the conversation (up to 512 tokens)
  python tools/bitnet/chat.py -p "What is the capital of France?"

The PC only tokenizes and detokenizes; the whole model runs on the board.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import HF_DIR, STOP_IDS  # noqa: E402

BAUD = 115200


class Board:
    def __init__(self, port: str, show_logs: bool):
        from serial_port import open_port

        self.ser = open_port(port, BAUD, 1)
        self.show_logs = show_logs

    def line(self, timeout: float = 600.0) -> str:
        end = time.time() + timeout
        buf = b""
        while time.time() < end:
            c = self.ser.readline()
            if not c:
                continue
            buf += c
            if buf.endswith(b"\n"):
                s = buf.decode(errors="replace").strip()
                buf = b""
                if s.startswith("#"):
                    if self.show_logs:
                        print(f"\033[2m{s}\033[0m", file=sys.stderr)
                    continue
                if s:
                    return s
        raise TimeoutError("no reply from board (is bitnet_zybo running? see STATUS.md)")

    def send(self, s: str) -> None:
        self.ser.write((s + "\n").encode())

    def wait_ready(self) -> None:
        self.ser.reset_input_buffer()
        self.send("I")
        while True:
            s = self.line(timeout=300)
            if s.startswith("I ") or s == "READY":
                return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("-p", "--prompt", help="ask one question and exit")
    ap.add_argument("-n", "--max-tokens", type=int, default=64)
    ap.add_argument("--multi", action="store_true", help="multi-turn: keep context between questions")
    ap.add_argument("--logs", action="store_true", help="show board log lines")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    b = Board(args.port, args.logs)
    print("connecting to board...", end=" ", flush=True)
    b.wait_ready()
    print("ready.")

    pending: list[int] = []   # last generated token, not yet fed through the model
    first = True
    while True:
        q = args.prompt if args.prompt else input("\nyou> ").strip()
        if not q:
            continue
        if q in ("/quit", "/exit"):
            return 0
        if not args.multi or first or q == "/reset":
            b.send("R")
            b.line()
            pending, first = [], False
            if q == "/reset":
                continue
            text = f"<|begin_of_text|>User: {q}<|eot_id|>Assistant: "
        else:
            text = f"User: {q}<|eot_id|>Assistant: "
        ids = pending + tok.encode(text, add_special_tokens=False).ids
        b.send("G " + " ".join(str(i) for i in [args.max_tokens] + ids))

        out: list[int] = []
        shown = ""
        t_first = None
        t0 = time.time()
        print("bitnet> ", end="", flush=True)
        while True:
            s = b.line()
            if s.startswith("X"):
                print(f"\n[board error] {s[1:].strip()}")
                break
            if s.startswith("T "):
                tid = int(s.split()[1])
                if t_first is None:
                    t_first = time.time()
                pending = [tid]
                if tid in STOP_IDS:
                    continue
                out.append(tid)
                text_now = tok.decode(out)
                print(text_now[len(shown):], end="", flush=True)
                shown = text_now
            elif s.startswith("E "):
                n_prompt, prefill_ms, n_gen, decode_ms = (int(v) for v in s.split()[1:5])
                steps = max(n_gen - 1, 1)
                tps = steps * 1000.0 / decode_ms if decode_ms else 0.0
                print(f"\n\033[2m[{n_prompt} prompt tokens in {prefill_ms / 1000:.1f}s, "
                      f"{n_gen} tokens, decode {tps:.2f} tok/s]\033[0m")
                break
        if args.prompt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
