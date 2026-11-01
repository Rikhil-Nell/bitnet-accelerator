"""Send one line to the board firmware and print replies until a regex matches.

    python tools/bitnet/board_cmd.py "I" --until "^I "
    python tools/bitnet/board_cmd.py --prompt "What is the capital of France?" --max 32
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import serial


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("line", nargs="?", default="")
    ap.add_argument("--prompt", help="chat-format and tokenize this prompt, send as G")
    ap.add_argument("--max", type=int, default=32)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--until", default=r"^(E |OK|X |I |READY)")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

    line, tok = args.line, None
    if args.prompt:
        from tokenizers import Tokenizer
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from generate import HF_DIR  # noqa: E402
        tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
        ids = tok.encode(f"<|begin_of_text|>User: {args.prompt}<|eot_id|>Assistant: ",
                         add_special_tokens=False).ids
        line = "R\nG %d %s" % (args.max, " ".join(map(str, ids)))
        args.until = r"^(E |X )"

    ser = serial.Serial(args.port, 115200, timeout=0.5)
    ser.reset_input_buffer()
    if line:
        ser.write((line + "\n").encode())
    out_ids, deadline, pat = [], time.time() + args.timeout, re.compile(args.until)
    buf = b""
    while time.time() < deadline:
        buf += ser.read(4096)
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            s = raw.decode(errors="replace").rstrip("\r")
            print(s, flush=True)
            if s.startswith("T "):
                out_ids.append(int(s.split()[1]))
            if pat.search(s):
                if tok is not None:
                    print("TEXT:", tok.decode(out_ids))
                return 0
    print("[board_cmd] timeout")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
