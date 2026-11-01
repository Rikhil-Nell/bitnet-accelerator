# Web UI

SolidJS + Vite. The write-up (`src/Writeup.tsx`, numbers in `src/data.ts`) and the live
demo (`src/Demo.tsx`), served by `web/server.py`, which owns the board's UART.

    make demo            # build + serve with the board on COM4 -> http://127.0.0.1:8000
    make demo-replay     # no board
    npm run dev          # in web/ui, with web/server.py running: hot reload on :5173
