#!/usr/bin/env python3
"""Bigger, more varied corpus for the n-gram draft table: the model writes its own
questions (sampled, one per topic and style), then answers them greedily, exactly as
the board would. Appends to $BITNET_WORK/spec_corpus2.npz.

  python tools/bitnet/spec_corpus2.py --rounds 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_coarse_head import PROMPTS as HELD_OUT  # noqa: E402
from generate import HF_DIR  # noqa: E402
from gpu_model import GpuBitNet, chat_prompt  # noqa: E402

from paths import WORK  # noqa: E402

OUT = WORK / "spec_corpus2.npz"
TOPICS = """animals plants oceans rivers mountains deserts weather climate volcanoes earthquakes planets stars
galaxies the moon the sun gravity light sound electricity magnetism atoms molecules chemistry cells DNA genetics
evolution the human body the heart the brain bones muscles nutrition vitamins sleep exercise diseases vaccines
medicine history ancient Egypt ancient Rome ancient Greece the Middle Ages the Renaissance World War II the Cold War
inventions computers the internet programming algorithms databases networks cybersecurity artificial intelligence
robots electronics transistors semiconductors batteries solar power wind power engines cars airplanes trains ships
rockets space exploration satellites mathematics geometry algebra prime numbers fractions statistics probability
economics money banks inflation trade companies marketing jobs education schools universities languages English
grammar writing poetry novels Shakespeare music instruments painting sculpture architecture movies photography
sports football basketball tennis the Olympics chess cooking bread coffee tea fruits vegetables farming forests
recycling pollution energy countries capitals cities Europe Asia Africa South America North America Australia
Antarctica famous scientists famous inventors famous painters philosophy psychology emotions friendship
communication time calendars clocks maps geography flags currencies holidays traditions transportation bridges
buildings materials metals plastics glass water fire air soil insects birds fish mammals reptiles dinosaurs""".split()
STYLES = ["Write one short factual question about {t}. Only write the question.",
          "Ask a simple question a student might ask about {t}. Only write the question.",
          "Write a short 'what is' or 'how does' question about {t}. Only write the question."]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    m = GpuBitNet()
    held = {q.strip().lower() for q in HELD_OUT}
    seqs, seen = [], set()
    if OUT.exists():
        d = np.load(OUT)
        o = 0
        for L in d["lens"].tolist():
            seqs.append(d["ids"][o : o + L].tolist())
            o += L
    jobs = [(st.format(t=t)) for _ in range(args.rounds) for t in TOPICS for st in STYLES]
    for b0 in range(0, len(jobs), args.batch):
        batch = jobs[b0 : b0 + args.batch]
        qs_ids = m.generate([tok.encode(chat_prompt(j), add_special_tokens=False).ids for j in batch],
                            max_new=32, temperature=0.9, seed=len(seqs) + b0)
        qs = []
        for ids in qs_ids:
            q = tok.decode(ids).strip().split("\n")[0].strip().strip('"')
            if q.endswith("?") and 10 < len(q) < 160 and q.lower() not in held and q not in seen:
                seen.add(q)
                qs.append(q)
        if not qs:
            continue
        prompts = [tok.encode(chat_prompt(q), add_special_tokens=False).ids for q in qs]
        outs = m.generate(prompts, max_new=64)
        for p, o in zip(prompts, outs):
            seqs.append(p + o)
        print(f"{b0 + len(batch):5d}/{len(jobs)} +{len(qs)}  {qs[0]!r} -> {tok.decode(outs[0])[:60]!r}", flush=True)
        if len(seqs) % 200 < len(qs):
            np.savez(OUT, ids=np.concatenate([np.array(s) for s in seqs]), lens=np.array([len(s) for s in seqs]))
    np.savez(OUT, ids=np.concatenate([np.array(s) for s in seqs]), lens=np.array([len(s) for s in seqs]))
    print(f"saved {len(seqs)} sequences, {sum(map(len, seqs))} tokens to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
