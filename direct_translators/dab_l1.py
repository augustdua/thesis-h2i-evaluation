"""DAB-L1 on saved translator images.

score_dir is the routine from _scratch/compute_thesis_diag_a5_vib.py.
Tissue mask: H&E mean optical density > 0.15. DAB-positive: target DAB > 0.3.
StainDeconvDAB lives in src/dab.py.

  python -u direct_translators/dab_l1.py
  python -u direct_translators/dab_l1.py --gen-dir path --split val --name my_row
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve  # noqa: E402
from src.dab import StainDeconvDAB, score_dir  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--her2", default=resolve(ROOT, cfg.get("her2_root") or ""))
    ap.add_argument("--gen-dir", default="")
    ap.add_argument("--split", default="val")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "dab_l1.json"))
    args = ap.parse_args()
    if not args.her2:
        raise SystemExit("Set her2_root in configs/chapter4.json")
    dab = StainDeconvDAB()
    data = Path(args.her2)
    rows = {}
    if args.gen_dir:
        rows[args.name] = score_dir(Path(args.gen_dir), data, args.split, dab)
    else:
        for name, raw in (cfg.get("gen_dirs") or {}).items():
            if not raw:
                rows[name] = {"status": "not_computed", "reason": "gen dir not set"}
                continue
            rows[name] = score_dir(Path(resolve(ROOT, raw)), data, args.split, dab)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "definition": {
            "tissue_mask": "HE mean OD > 0.15",
            "dab_positive": "target DAB > 0.3",
            "dab": "Ruifrok StainDeconvDAB from compute_thesis_diag_a5_vib.py",
        },
        "split": args.split,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
