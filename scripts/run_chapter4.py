"""Outline for the Chapter 4 numbers. Writes under outputs/. Does not start Modal.

  python -u scripts/run_chapter4.py frequency
  python -u scripts/run_chapter4.py exemplar-baselines -- --limit 8 --const-idx 0
  python -u scripts/run_chapter4.py list
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CMDS = {
    "exemplar-baselines": ROOT / "exemplar" / "eval_baselines.py",
    "exemplar-k4": ROOT / "exemplar" / "eval_k4.py",
    "cka-rank": ROOT / "direct_translators" / "cka_spatial_rank.py",
    "frequency": ROOT / "direct_translators" / "hf_study.py",
    "dab-l1": ROOT / "direct_translators" / "dab_l1.py",
    "vib": ROOT / "vib" / "score_vib.py",
}


def main() -> None:
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "list"}:
        print("commands:", ", ".join(CMDS))
        print("extra arguments after -- are passed to the script")
        return
    name = sys.argv[1]
    if name not in CMDS:
        raise SystemExit(f"unknown command {name}. choices: {', '.join(CMDS)}")
    extra = sys.argv[2:]
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd = [sys.executable, "-u", str(CMDS[name]), *extra]
    print("RUN", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
