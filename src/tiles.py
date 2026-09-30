"""Slide-spread tile pick. Copied from _scratch/eval_c9_spatial_reff.py."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path


def slide_id(path: str) -> str:
    return Path(path).name.split("_")[0]


def pick_slide_spread(files: list[str], n: int, seed: int) -> list[str]:
    by: dict[str, list[str]] = defaultdict(list)
    for p in files:
        by[slide_id(p)].append(p)
    slides = sorted(by)
    rng = random.Random(seed)
    for s in slides:
        rng.shuffle(by[s])
    picked: list[str] = []
    i = 0
    while len(picked) < n:
        progress = False
        for s in slides:
            if i < len(by[s]):
                picked.append(by[s][i])
                progress = True
                if len(picked) >= n:
                    break
        if not progress:
            break
        i += 1
    return sorted(picked)
