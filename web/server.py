#!/usr/bin/env python3
"""Web front end for the board: chat + live profile, plus the write-up.

  python web/server.py                 # board on COM4, http://127.0.0.1:8000
  python web/server.py --replay        # no board: replays recorded answers (UI work, offline demo)

The board firmware (fpga/ps/bitnet_zybo) speaks a line protocol on the UART; this
server owns the port, tokenizes / detokenizes, and streams events to the browser
over a WebSocket:

  {"type": "token",   "id": 791, "text": "The", "ms": 172}
  {"type": "profile", "phase": "decode", "us": {"PL_wait": 118000, ...}}
  {"type": "cand",    "ids": [...], "logits": [...], "texts": [...]}     (top candidates, if traced)
  {"type": "done",    "n_prompt": 14, "prefill_ms": 2266, "n_gen": 8, "decode_ms": 1366}
  {"type": "error",   "message": "..."}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "bitnet"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

DIST = Path(__file__).resolve().parent / "ui" / "dist"
STOP_IDS = {128001, 128009}


def load_tokenizer():
    from tokenizers import Tokenizer
    from generate import HF_DIR
    return Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))


def parse_profile(line: str) -> tuple[str, dict[str, int]] | None:
    # "# profile decode, us per token: embed=38 rmsnorm=5234 ... total=195365"
    if not line.startswith("# profile"):
        return None
    head, _, body = line.partition(":")
    phase = head.split()[2].rstrip(",")
    us = {}
    for kv in body.split():
        k, _, v = kv.partition("=")
        if v.isdigit():
            us[k] = int(v)
    return phase, us


class Board:
    """Serial link to bitnet_zybo. One generation at a time."""

    def __init__(self, port: str, baud: int):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.2)
        self.lock = threading.Lock()
        self.buf = b""

    def _line(self, deadline: float) -> str | None:
        while time.time() < deadline:
            if b"\n" in self.buf:
                raw, self.buf = self.buf.split(b"\n", 1)
                return raw.decode(errors="replace").strip()
            self.buf += self.ser.read(4096)
        return None

    def command(self, s: str, until: str, timeout: float = 10.0) -> list[str]:
        self.ser.write((s + "\n").encode())
        out, end = [], time.time() + timeout
        while True:
            ln = self._line(end)
            if ln is None:
                raise TimeoutError(f"no reply to {s!r}")
            out.append(ln)
            if ln.startswith(until):
                return out

    def generate(self, ids: list[int], max_new: int, emit) -> None:
        """Send G and call emit(line) for every reply line until the E line."""
        self.ser.write(("G " + " ".join(map(str, [max_new] + ids)) + "\n").encode())
        end = time.time() + 600
        while True:
            ln = self._line(end)
            if ln is None:
                raise TimeoutError("board stopped answering")
            emit(ln)
            if ln.startswith("E ") or ln.startswith("X"):
                return


class Session:
    """Turns board lines into browser events (shared by the board and the replay)."""

    def __init__(self, tok, send):
        self.tok, self.send = tok, send
        self.out: list[int] = []
        self.shown = ""

    def line(self, ln: str) -> None:
        if ln.startswith("T "):
            parts = ln.split()
            tid, ms = int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
            if tid in STOP_IDS:
                return
            self.out.append(tid)
            text = self.tok.decode(self.out)
            self.send({"type": "token", "id": tid, "text": text[len(self.shown):], "ms": ms})
            self.shown = text
        elif ln.startswith("C "):
            pairs = [p.split(":") for p in ln.split()[1:]]
            ids = [int(a) for a, _ in pairs]
            self.send({"type": "cand", "ids": ids, "logits": [float(b) for _, b in pairs],
                       "texts": [self.tok.decode([i]) for i in ids]})
        elif ln.startswith("E "):
            n_prompt, prefill_ms, n_gen, decode_ms = (int(v) for v in ln.split()[1:5])
            self.send({"type": "done", "n_prompt": n_prompt, "prefill_ms": prefill_ms,
                       "n_gen": n_gen, "decode_ms": decode_ms})
        elif ln.startswith("X"):
            self.send({"type": "error", "message": ln[1:].strip()})
        else:
            p = parse_profile(ln)
            if p:
                self.send({"type": "profile", "phase": p[0], "us": p[1]})


REPLAY = [  # recorded on the board (docs/results), used by --replay
    ("The capital of France is Paris.",
     "# profile decode, us per token: embed=38 rmsnorm=5234 quantize=4689 flush_x=295 PL_wait=153023 "
     "inval_y=2033 dequant=2435 rope=902 attention=21736 elementwise=2561 head_argmax=2414 total=195365"),
]


def make_app(board: Board | None, tok) -> FastAPI:
    app = FastAPI()
    gen_lock = asyncio.Lock()

    @app.get("/api/status")
    def status():
        if board is None:
            return {"board": False, "mode": "replay"}
        try:
            with board.lock:
                info = board.command("I", "I ", timeout=3)[-1]
            return {"board": True, "info": info}
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"board": False, "error": str(e)}, status_code=503)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        loop = asyncio.get_running_loop()
        first = True
        pending: list[int] = []
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                if msg.get("type") != "ask":
                    continue
                q, multi, max_new = msg["prompt"].strip(), bool(msg.get("multi")), int(msg.get("max", 64))
                queue: asyncio.Queue = asyncio.Queue()

                def send(ev, queue=queue):
                    loop.call_soon_threadsafe(queue.put_nowait, ev)

                sess = Session(tok, send)

                def work(q=q, multi=multi, max_new=max_new, sess=sess):
                    nonlocal first, pending
                    try:
                        if board is None:
                            text, prof = REPLAY[0]
                            for i, piece in enumerate(tok.encode(text, add_special_tokens=False).ids):
                                time.sleep(0.15)
                                sess.line(f"T {piece} 150")
                            sess.line(prof)
                            sess.line(f"E 14 2266 {len(sess.out)} {150 * len(sess.out)}")
                            return
                        with board.lock:
                            if not multi or first:
                                board.command("R", "OK")
                                pending = []
                                text = f"<|begin_of_text|>User: {q}<|eot_id|>Assistant: "
                            else:
                                text = f"User: {q}<|eot_id|>Assistant: "
                            first = False
                            ids = pending + tok.encode(text, add_special_tokens=False).ids
                            board.generate(ids, max_new, sess.line)
                            pending = sess.out[-1:] if sess.out else []
                    except Exception as e:  # noqa: BLE001
                        send({"type": "error", "message": str(e)})
                    finally:
                        send(None)

                async with gen_lock:
                    loop.run_in_executor(None, work)
                    while (ev := await queue.get()) is not None:
                        await sock.send_text(json.dumps(ev))
        except WebSocketDisconnect:
            return

    if DIST.exists():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        def index(path: str):
            f = DIST / path
            return FileResponse(f if path and f.is_file() else DIST / "index.html")
    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--replay", action="store_true", help="no board: replay a recorded answer")
    ap.add_argument("--http", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    tok = load_tokenizer()
    board = None if args.replay else Board(args.port, args.baud)
    if board is not None:   # ask the firmware for top-5 candidates per token ('C' toggles)
        try:
            for _ in range(2):
                if "cands=1" in board.command("C", "OK cands=", timeout=3)[-1]:
                    break
        except Exception as e:  # noqa: BLE001
            print(f"note: candidate trace not available ({e})")
    print(f"http://127.0.0.1:{args.http}  ({'replay' if board is None else args.port})")
    uvicorn.run(make_app(board, tok), host="127.0.0.1", port=args.http, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
