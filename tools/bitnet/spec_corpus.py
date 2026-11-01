#!/usr/bin/env python3
"""Build a corpus of the model's own greedy answers (for the n-gram draft table).

Runs the board model on the GPU (gpu_model.py), batched.

  python tools/bitnet/spec_corpus.py            # writes D:/xilinx-tools/spec_corpus.npz

Questions are generated from templates; the 12 evaluation prompts in
eval_coarse_head.py are excluded so the table is judged on unseen questions.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_coarse_head import PROMPTS as HELD_OUT  # noqa: E402
from generate import HF_DIR  # noqa: E402

OUT = Path("D:/xilinx-tools/spec_corpus.npz")

COUNTRIES = ["Japan", "Germany", "Brazil", "Canada", "Australia", "India", "Egypt", "Kenya", "Mexico", "Italy",
             "Spain", "Norway", "Argentina", "China", "Russia", "Peru", "Chile", "Sweden", "Greece", "Turkey",
             "Nigeria", "Vietnam", "Thailand", "Poland", "Portugal", "Ireland", "Cuba", "Iran", "Iraq", "Nepal"]
ANIMALS = ["an octopus", "a cat", "a dog", "a horse", "an ant", "a bee", "a crab", "a snake", "a bird",
           "a starfish", "a frog", "a butterfly", "a cow", "an elephant", "a whale", "a shark", "a bat", "a penguin"]
THINGS = ["photosynthesis", "gravity", "a black hole", "DNA", "the internet", "a transistor", "an FPGA",
          "electricity", "a volcano", "an earthquake", "the water cycle", "inflation", "democracy", "a vaccine",
          "machine learning", "a neural network", "the moon", "the sun", "a rainbow", "climate change",
          "a computer", "a battery", "a solar panel", "an atom", "a cell", "evolution", "the stock market",
          "a hurricane", "a glacier", "an operating system", "a compiler", "a database", "encryption",
          "the immune system", "a semiconductor", "radio waves", "magnetism", "a laser", "a microscope", "oxygen"]
PEOPLE = ["Albert Einstein", "Isaac Newton", "Marie Curie", "Leonardo da Vinci", "Charles Darwin",
          "Nikola Tesla", "Ada Lovelace", "Alan Turing", "Galileo Galilei", "William Shakespeare",
          "Mahatma Gandhi", "Abraham Lincoln", "Cleopatra", "Napoleon", "Mozart", "Beethoven", "Picasso",
          "Thomas Edison", "Aristotle", "Confucius"]
TEMPLATES_COUNTRY = ["What is the capital of {}?", "What language is spoken in {}?", "What continent is {} in?",
                     "What is the currency of {}?", "Tell me one interesting fact about {}."]
TEMPLATES_ANIMAL = ["How many legs does {} have?", "What does {} eat?", "Where does {} live?"]
TEMPLATES_THING = ["What is {}?", "Explain {} in simple terms.", "Why is {} important?", "How does {} work?",
                   "Explain in two sentences what {} is."]
TEMPLATES_PERSON = ["Who was {}?", "What is {} famous for?", "When did {} live?"]
MATH = ["What is {} times {}?", "What is {} plus {}?", "What is {} minus {}?"]
GENERAL = ["Give me three tips for studying.", "How do I make a cup of tea?", "What is the tallest mountain?",
           "What is the speed of light?", "How many days are in a year?", "What is the largest planet?",
           "Who invented the telephone?", "What is the chemical symbol for gold?", "How many continents are there?",
           "What is the smallest prime number?", "Give me three tips for sleeping better.",
           "What is the freezing point of water in Fahrenheit?", "How far is the Moon from Earth?",
           "What is the longest river in the world?", "Name three primary colors.",
           "What are the three states of matter?", "How many bones are in the human body?",
           "What is the hardest natural substance?", "Which planet is closest to the Sun?",
           "What do bees make?", "What is the main ingredient in bread?", "Give me three tips for saving money."]


def questions() -> list[str]:
    q = [t.format(c) for c in COUNTRIES for t in TEMPLATES_COUNTRY]
    q += [t.format(a) for a in ANIMALS for t in TEMPLATES_ANIMAL]
    q += [t.format(x) for x in THINGS for t in TEMPLATES_THING]
    q += [t.format(p) for p in PEOPLE for t in TEMPLATES_PERSON]
    rng = random.Random(0)
    q += [rng.choice(MATH).format(rng.randint(2, 30), rng.randint(2, 30)) for _ in range(60)]
    q += GENERAL
    held = set(HELD_OUT)
    return [x for x in q if x not in held]


def main() -> int:
    from tokenizers import Tokenizer
    from gpu_model import GpuBitNet, chat_prompt
    tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    qs = questions()
    print(f"{len(qs)} questions", flush=True)
    m = GpuBitNet()
    seqs: list[list[int]] = []
    for b0 in range(0, len(qs), 48):
        batch = qs[b0 : b0 + 48]
        prompts = [tok.encode(chat_prompt(q), add_special_tokens=False).ids for q in batch]
        outs = m.generate(prompts, max_new=64)
        for q, p, o in zip(batch, prompts, outs):
            seqs.append(p + o)
        print(f"{b0 + len(batch):4d}/{len(qs)}  {batch[0]!r} -> {tok.decode(outs[0])[:70]!r}", flush=True)
    np.savez(OUT, ids=np.concatenate([np.array(s) for s in seqs]), lens=np.array([len(s) for s in seqs]))
    print(f"saved {len(seqs)} sequences, {sum(map(len, seqs))} tokens to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
