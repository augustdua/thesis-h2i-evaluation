"""High-frequency content of real IHC and of direct-translator outputs.

Measurement loop from _scratch/ch4_figs/hf_study.py. Definitions of band
energy, half-Nyquist power, low-pass, and numpy DAB are in src/frequency.py.
Writes outputs/hf_study.json. The thesis figure redraw is not included.

  python -u direct_translators/hf_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve  # noqa: E402
from src.frequency import (  # noqa: E402
    band_fractions,
    dab,
    i_hf,
    load,
    lowpass_rgb,
    luma,
    summarise,
)

CELLS = [
    ("xdec_c64", "Cross-decode, C=64"),
    ("xdec_c512", "Cross-decode, C=512"),
    ("patchnce_c512", "+ Pair PatchNCE, C=512"),
    ("pairnce_c512", "+ PairNCE, C=512"),
    ("pairnce_c64", "+ PairNCE, C=64"),
    ("patchnce_c64", "+ Pair PatchNCE, C=64"),
    ("patchnce_c128", "+ Pair PatchNCE, C=128"),
    ("patchnce_c256", "+ Pair PatchNCE, C=256"),
    ("vib_c64_b1e-4", "VIB, C_Z=64, beta=1e-4"),
]


def _gen_dir(cfg: dict, cell: str) -> Path | None:
    raw = (cfg.get("gen_dirs") or {}).get(cell) or ""
    if raw:
        return Path(resolve(ROOT, raw))
    root = cfg.get("gen_root") or ""
    if root:
        return Path(resolve(ROOT, root)) / cell
    return None


def main() -> None:
    cfg = load_config(ROOT)
    her2 = cfg.get("her2_root") or ""
    if not her2:
        raise SystemExit("Set her2_root in configs/chapter4.json")
    data = Path(resolve(ROOT, her2))
    stems_path = Path(resolve(ROOT, cfg.get("hf_stems") or "configs/hf_study_stems.json"))
    stems = [s for s, _ in json.loads(stems_path.read_text(encoding="utf-8"))]
    res: dict = {
        "n_tiles": len(stems),
        "levels": "level 1 = finest (2-4 px periods at 256 px)",
        "stems": str(stems_path),
        "cells": {},
    }

    real_rows, dab_rows = [], []
    for s in stems:
        rgb = load(data / "IHC" / "val" / f"{s}.jpg")
        y = luma(rgb)
        real_rows.append({"bands": band_fractions(y).tolist(), "i_hf": i_hf(y)})
        d0 = dab(rgb)
        row = {"bf0": float((d0 > 0.3).mean()), "dab_mean0": float(d0.mean())}
        for k in (1, 2, 3):
            dk = dab(lowpass_rgb(rgb, k))
            row[f"bf{k}"] = float((dk > 0.3).mean())
            row[f"dab_l1_{k}"] = float(np.abs(dk - d0).mean())
        dab_rows.append(row)
    res["real_ihc"] = summarise(real_rows)

    pos = [r for r in dab_rows if r["bf0"] >= 0.05]
    lp = {
        "n_all": len(dab_rows),
        "n_dab_pos": len(pos),
        "dab_pos_rule": "IHC brown fraction >= 0.05",
    }
    for k in (1, 2, 3):
        lp[f"LL{k}"] = {
            "grid_equivalent": f"{256 // 2 ** k}x{256 // 2 ** k}",
            "brown_frac_ratio_all": float(
                np.sum([r[f"bf{k}"] for r in dab_rows])
                / max(np.sum([r["bf0"] for r in dab_rows]), 1e-12)
            ),
            "brown_frac_ratio_dab_pos": float(
                np.sum([r[f"bf{k}"] for r in pos])
                / max(np.sum([r["bf0"] for r in pos]), 1e-12)
            ),
            "dab_l1_all": float(np.mean([r[f"dab_l1_{k}"] for r in dab_rows])),
            "dab_l1_dab_pos": float(np.mean([r[f"dab_l1_{k}"] for r in pos])) if pos else None,
        }
    res["ihc_lowpass_dab"] = lp

    for cell, label in CELLS:
        d = _gen_dir(cfg, cell)
        if d is None or not d.is_dir():
            res["cells"][cell] = {
                "status": "not_computed",
                "label": label,
                "reason": "gen dir not set or missing",
            }
            continue
        rows = []
        for s in stems:
            p = d / f"{s}.png"
            if p.is_file():
                y = luma(load(p))
                rows.append({"bands": band_fractions(y).tolist(), "i_hf": i_hf(y)})
        if len(rows) != len(stems):
            print(f"WARN {cell} {len(rows)}/{len(stems)}")
            res["cells"][cell] = {
                "status": "not_computed",
                "label": label,
                "n": len(rows),
                "reason": "incomplete png set for the 240 stems",
            }
            continue
        sm = summarise(rows)
        sm["label"] = label
        sm["status"] = "computed"
        sm["band_ratio_to_real"] = (
            np.array(sm["band_frac_mean"]) / np.array(res["real_ihc"]["band_frac_mean"])
        ).tolist()
        sm["i_hf_ratio_to_real"] = sm["i_hf_mean"] / res["real_ihc"]["i_hf_mean"]
        res["cells"][cell] = sm

    out = ROOT / "outputs" / "hf_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
